#!/usr/bin/env python3
"""Collect explicitly unit-labelled ASIC results; never use ORFS vectorless power.

Usage: python report_metrics.py WORK_DIR [--rtl-manifest MANIFEST.json] [--out metrics.json]
Missing stages produce null fields. Stale or unverified power reports fail closed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def collect(work, manifest_path=None):
    work = Path(work).resolve()
    config = (work / 'config.mk').read_text()
    match = re.search(r'^export DESIGN_NAME\s*=\s*(\S+)\s*$', config, re.M)
    if not match:
        raise ValueError('Missing DESIGN_NAME in config.mk')
    top = match.group(1)
    result_dir = work / 'results' / 'nangate45' / top / 'base'
    summary = work / 'summary.txt'
    values = {}
    if summary.exists():
        for line in summary.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                values[parts[0]] = parts[1]

    def number(key):
        value = values.get(key)
        if value in (None, 'None', 'null', '-'):
            return None
        value = float(value)
        if not math.isfinite(value):
            raise ValueError('Non-finite result: ' + key)
        return value

    marker = work / 'route_complete.json'
    finished = marker.exists() and summary.exists() and all((result_dir / name).is_file() and
        (result_dir / name).stat().st_size > 0 for name in ('6_final.odb', '6_final.v', '6_final.spef'))
    if finished:
        completion = json.loads(marker.read_text())
        if (completion.get('ok') is not True or completion.get('top') != top
                or completion.get('summary_sha256') != sha256(summary)
                or completion.get('netlist_sha256') != sha256(result_dir / '6_final.v')):
            raise ValueError('Stale or invalid route completion proof')
    physical = {name: number(key) if finished else None for name, key in {
        'cell_area_um2': 'cell_area_um2', 'core_area_um2': 'core_area_um2',
        'die_area_um2': 'die_area_um2', 'setup_slack_ns': 'wns_ns',
        'hold_slack_ns': 'hold_ws_ns', 'setup_tns_ns': 'tns_ns',
        'drc_violations': 'drc_violations', 'instance_count': 'instance_count',
    }.items()}
    for stem in ('cell', 'core', 'die'):
        value = physical[stem + '_area_um2']
        physical[stem + '_area_mm2'] = None if value is None else value / 1e6
    report = dict(design=work.name, top=top, physical_stage='routed' if finished else 'incomplete',
        technology='Nangate45 typical, 1.10 V, 25 C', clock_ns=number('clock_ns'),
        routed=physical, rtl_timing=None, power=None,
        notes=['Cell area is summed standard-cell footprint; core/die include floorplan whitespace.',
               'RTL latency and event interval are not timing slack.',
               'No fabricated-chip or multi-corner signoff measurement is claimed.'])

    if manifest_path:
        manifest = json.loads(Path(manifest_path).read_text())
        if manifest.get('top') != top:
            raise ValueError('RTL manifest top differs from routed design')
        rtl = manifest.get('verification', {}).get('rtl', {})
        if not rtl.get('ok') or any(not rtl.get(mode, {}).get('ok') or
            rtl[mode].get('n_results', 0) <= 0 or
            rtl[mode].get('n_exact_events') != rtl[mode].get('n_results')
            for mode in ('seq', 'stream')):
            raise ValueError('RTL manifest lacks passing sequential/streaming checks')
        timing = manifest.get('measured_rtl_timing', {})
        cycles = manifest.get('latency_clk', [timing.get('latency_min'), timing.get('latency_max')])
        if isinstance(cycles, list) and len(cycles) == 2 and cycles[0] == cycles[1]:
            cycles = cycles[0]
        interval = manifest.get('ii', timing.get('ii_ready_max'))
        clock = report['clock_ns'] or manifest.get('clock_period_ns') or manifest.get('clock_ns')
        report['rtl_timing'] = dict(latency_cycles=cycles, interval_cycles=interval,
            latency_ns=cycles * clock if isinstance(cycles, (int, float)) and clock else None,
            interval_ns=interval * clock if isinstance(interval, (int, float)) and clock else None,
            clock_ns=clock, scope='verified RTL interface, not a measured maximum frequency',
            manifest_sha256=sha256(manifest_path))

    power_json = work / 'power' / 'power_vcd.json'
    if power_json.exists():
        if not finished:
            raise ValueError('Power exists without a complete routed result')
        power = json.loads(power_json.read_text())
        proof = json.loads((work / 'power' / 'gate_check.json').read_text())
        if power.get('method') != 'vcd' or power.get('valid_for_comparison') is not True:
            raise ValueError('Only validated VCD power is accepted')
        if not proof.get('ok') or proof.get('events', 0) <= 0 or proof.get('stream_exact') != proof['events'] or proof.get('t0') != 0:
            raise ValueError('Missing exact routed-gate/t0 proof')
        for field, path in [('netlist_sha256', result_dir / '6_final.v'),
                            ('vcd_sha256', work / 'power' / 'activity.vcd')]:
            digest = sha256(path)
            if power.get(field) != digest or proof.get(field) != digest:
                raise ValueError('Stale power or gate proof: ' + field)
        power_rpt = work / 'power' / 'power_vcd.rpt'
        if power.get('power_report_sha256') != sha256(power_rpt) or 'STA-1452' in power_rpt.read_text():
            raise ValueError('Stale or invalid power report')
        if report['clock_ns'] is not None and not math.isclose(float(power['clock_ns']), report['clock_ns']):
            raise ValueError('Power and physical reports use different clocks')
        for key in ('total_mW', 'internal_mW', 'switching_mW', 'leakage_mW'):
            if not math.isfinite(float(power[key])) or float(power[key]) < 0:
                raise ValueError('Invalid power value: ' + key)
        report['power'] = {key: power[key] for key in ('total_mW', 'internal_mW', 'switching_mW', 'leakage_mW')}
        report['power'].update(method='corrected t0 VCD, zero-delay routed-gate activity',
                              gate_events=proof['events'], source_sha256=sha256(power_rpt))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work_dir', type=Path)
    parser.add_argument('--rtl-manifest', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = json.dumps(collect(args.work_dir, args.rtl_manifest), indent=2, allow_nan=False) + '\n'
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result)
    print(result, end='')
