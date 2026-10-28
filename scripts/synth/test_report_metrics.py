"""Unit conversions, missing-stage handling and stale-power rejection."""
import json
from pathlib import Path
import tempfile
from report_metrics import collect, sha256


def main():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        (root / 'config.mk').write_text('export DESIGN_NAME = test_core\n')
        assert collect(root)['routed']['cell_area_mm2'] is None
        results = root / 'results/nangate45/test_core/base'
        results.mkdir(parents=True)
        for name in ('6_final.odb', '6_final.v', '6_final.spef'):
            (results / name).write_text('fixture\n')
        (root / 'summary.txt').write_text('clock_ns 25\ncell_area_um2 420000\ncore_area_um2 1000000\nwns_ns 12.5\npower_total_W 1000\n')
        assert collect(root)['physical_stage'] == 'incomplete'
        completion = dict(ok=True, top='test_core', summary_sha256=sha256(root/'summary.txt'),
                          netlist_sha256=sha256(results/'6_final.v'))
        (root/'route_complete.json').write_text(json.dumps(completion))
        report = collect(root)
        assert report['routed']['cell_area_mm2'] == 0.42
        assert report['routed']['setup_slack_ns'] == 12.5
        assert report['power'] is None  # Bogus vectorless power never leaks into accepted metrics.
        manifest = dict(top='test_core', latency_clk=[2,2], ii=1,
            verification=dict(rtl=dict(ok=True, **{m:dict(ok=True,n_results=200,n_exact_events=200) for m in ('seq','stream')})))
        mp = root/'MANIFEST.json'
        mp.write_text(json.dumps(manifest))
        timing = collect(root,mp)['rtl_timing']
        assert timing['latency_ns']==50 and timing['interval_ns']==25
        manifest['verification']['rtl']['seq']['ok']=False
        mp.write_text(json.dumps(manifest))
        try:
            collect(root,mp)
        except ValueError as error:
            assert 'sequential/streaming' in str(error)
        else:
            raise AssertionError('Failing RTL manifest must be rejected')
        power_dir = root / 'power'
        power_dir.mkdir()
        (power_dir / 'activity.vcd').write_text('$enddefinitions $end\n#0\n0a\n#25\n1a\n')
        (power_dir / 'power_vcd.rpt').write_text('fixture corrected power report\n')
        hashes = dict(netlist_sha256=sha256(results / '6_final.v'), vcd_sha256=sha256(power_dir / 'activity.vcd'))
        (power_dir / 'gate_check.json').write_text(json.dumps(dict(ok=True, events=200, stream_exact=200, t0=0, **hashes)))
        power = dict(method='vcd', valid_for_comparison=True, clock_ns=25,
            total_mW=39, internal_mW=11, switching_mW=16, leakage_mW=12,
            power_report_sha256=sha256(power_dir / 'power_vcd.rpt'), **hashes)
        (power_dir / 'power_vcd.json').write_text(json.dumps(power))
        assert collect(root)['power']['total_mW'] == 39
        (results / '6_final.v').write_text('new netlist\n')
        try:
            collect(root)
        except ValueError as error:
            assert 'Stale' in str(error)
        else:
            raise AssertionError('Changed netlist must invalidate old power')
        completion['netlist_sha256'] = sha256(results/'6_final.v')
        (root/'route_complete.json').write_text(json.dumps(completion))
        try:
            collect(root)
        except ValueError as error:
            assert 'Stale power' in str(error)
        else:
            raise AssertionError('Updated route proof must not validate stale power')
    print('Metric units, missing stages, vectorless exclusion and stale-power rejection passed')


if __name__ == '__main__':
    main()
