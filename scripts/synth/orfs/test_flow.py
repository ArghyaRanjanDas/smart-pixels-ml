"""Bounded mocked-tool checks; does not invoke real synthesis or simulation."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('normalize',HERE/'normalize_vcd.py');normalizer=importlib.util.module_from_spec(spec);spec.loader.exec_module(normalizer)


class FlowTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.bin=self.root/'bin';self.bin.mkdir()
        self.work=self.root/'custom_work';self.res=self.work/'example/results/nangate45/demo/base';self.res.mkdir(parents=True)
        (self.work/'example/config.mk').write_text('export DESIGN_NAME = demo\n')
        for n in ['1_synth.v','6_final.v','6_final.odb']:(self.res/n).write_text('fixture\n')
        (self.res/'clock_period.txt').write_text('25000\n')
        self.ref=self.root/'verified_rtl';self.ref.mkdir()
        for n in ['rtl_seq.txt','rtl_stream.txt']:(self.ref/n).write_text('1 2 3 4\n'*200)
        (self.ref/'tb_demo.cpp').write_text('''#include "verilated.h"
static void tick() { top->eval(); }
int main() {
Verilated::commandArgs(argc, argv);
  // ---------- seq pass
  seq();
  // ---------- stream pass
  stream();
  dump("rtl_seq.txt", RS); dump("rtl_stream.txt", RT);
fopen("rtl_timing.json", "w");
}
''')
        cells=self.root/'cells.v';cells.write_text('module cells; endmodule\n')
        self.env=dict(os.environ,PATH=str(self.bin)+':'+os.environ['PATH'],PYTHON=sys.executable,
            WORK_ROOT=str(self.work),ORFS_HOME=str(self.root/'orfs'),RTL_TEST_DIR=str(self.ref),CELL_MODELS=str(cells))
        self.fake('verilator', '''from pathlib import Path
import os
out=Path('obj_dir');out.mkdir(exist_ok=True)
p=out/'simv'
p.write_text("#!"+os.environ['PYTHON']+"\\n"+''' + repr('''from pathlib import Path
import os
ref=Path(os.environ['RTL_TEST_DIR']);power=Path('tb_power.cpp').exists()
for name in (['pw_stream.txt'] if power else ['rtl_seq.txt','rtl_stream.txt']):
    text=(ref/('rtl_stream.txt' if power else name)).read_text()
    if os.environ.get('FAKE_MODE')=='bad':text=text.replace('1 2 3 4','9 9 9 9')
    if os.environ.get('FAKE_MODE')=='partial':text=''.join(text.splitlines(True)[:-1])
    Path(name).write_text(text)
if power:Path('activity.vcd').write_text('$timescale 1ps $end\\n#100\\n0a\\n#150\\n1a\\n')
''') + ''')
p.chmod(0o755)
''')
        self.fake('openroad', '''import os
print('PW_CLOCK_NS 25')
if os.environ.get('BAD_CLOCK'):print('[WARNING STA-1452] bad period')
for name in ['Total','Sequential','Combinational','Clock']:print(name+' 0.1 0.2 0.3 0.6 100.0%')
''')
    def fake(self,name,body):
        p=self.bin/name;p.write_text('#!'+sys.executable+'\n'+body);p.chmod(0o755)
    def tearDown(self):self.temp.cleanup()
    def invoke(self,name,**env):return subprocess.run(['bash',str(HERE/name),'example'],env={**self.env,**env},capture_output=True,text=True)
    def test_syntax_and_no_personal_paths(self):
        for p in HERE.glob('*.sh'):
            subprocess.run(['bash','-n',str(p)],check=True)
            self.assertNotIn('miniconda',p.read_text());self.assertNotIn('smartpix_hls',p.read_text())
    def test_gate_positive_negative_and_partial(self):
        self.assertEqual(self.invoke('gate_check_synth.sh').returncode,0)
        proof=json.loads((self.work/'example/gate/gate_check_synth.json').read_text());self.assertEqual(proof['stream_exact'],200);self.assertIn('netlist_sha256',proof)
        for mode in ['bad','partial']:
            result=self.invoke('gate_check_synth.sh',FAKE_MODE=mode)
            self.assertNotEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertFalse((self.work/'example/gate/gate_check_synth.json').exists())
    def test_power_proof_t0_warning_and_changed_netlist(self):
        result=self.invoke('power_vcd.sh');self.assertEqual(result.returncode,0,result.stderr)
        power=self.work/'example/power';proof=json.loads((power/'gate_check.json').read_text());self.assertEqual(proof['t0'],0)
        self.assertIn('#0\n',(power/'activity.vcd').read_text())
        result=self.invoke('power_report.sh');self.assertEqual(result.returncode,0,result.stderr)
        report=json.loads((power/'power_vcd.json').read_text());self.assertAlmostEqual(report['total_mW'],600.)
        self.assertEqual(report['netlist_sha256'],proof['netlist_sha256']);self.assertIn('power_report_sha256',report)
        self.assertNotEqual(self.invoke('power_report.sh',BAD_CLOCK='1').returncode,0)
        (self.res/'6_final.v').write_text('changed\n')
        self.assertNotEqual(self.invoke('power_report.sh').returncode,0)
    def test_route_tag_resume_and_completion_marker(self):
        self.fake('yosys','pass\n');self.fake('make','raise SystemExit(7)\n')
        flow=self.root/'orfs/flow';flow.mkdir(parents=True);(flow/'Makefile').write_text('# fixture\n')
        work=self.work/'example';src=work/'src';src.mkdir();rtl=src/'demo.v';rtl.write_text('module demo; endmodule\n')
        (work/'config.mk').write_text('export DESIGN_NAME = demo\nexport ABC_CLOCK_PERIOD_IN_PS = 25000\nexport CORE_UTILIZATION = 40\n')
        (work/'constraint.sdc').write_text('set clk_port_name ap_clk\n')
        def run(name,clock='25',port='ap_clk',util='40'):
            return subprocess.run(['bash',str(HERE/name),str(rtl),'demo',clock,port,util,'example'],env=self.env,capture_output=True,text=True)
        result=run('route_design.sh');self.assertNotEqual(result.returncode,0);self.assertIn('route_resume.sh',result.stderr)
        for name in ['summary.txt','route.done','route_complete.json']:(work/name).write_text('old result\n')
        for clock,port,util in [('26','ap_clk','40'),('25','wrong_clk','40'),('25','ap_clk','41')]:
            self.assertNotEqual(run('route_resume.sh',clock,port,util).returncode,0)
            self.assertTrue((work/'summary.txt').exists())
        result=run('route_resume.sh');self.assertEqual(result.returncode,7,result.stderr)
        for name in ['summary.txt','route.done','route_complete.json']:self.assertFalse((work/name).exists())
        (work/'summary.txt').write_text('cell_area_um2 123\n')
        result=subprocess.run(['bash','-c','source "$1"; WORK="$2"; TOP=demo; CLK_NS=25; CLK_PORT=ap_clk; UTIL=40; mark_route_complete',
            'fixture',str(HERE/'env.sh'),str(work)],env=self.env,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        proof=json.loads((work/'route_complete.json').read_text());self.assertTrue(proof['ok']);self.assertEqual(proof['clock_port'],'ap_clk')
        import hashlib
        self.assertEqual(proof['summary_sha256'],hashlib.sha256((work/'summary.txt').read_bytes()).hexdigest())
        self.assertEqual(proof['netlist_sha256'],hashlib.sha256((self.res/'6_final.v').read_bytes()).hexdigest())

    def test_power_mismatch_and_invalid_vcd(self):
        self.assertNotEqual(self.invoke('power_vcd.sh',FAKE_MODE='bad').returncode,0)
        self.assertFalse((self.work/'example/power/gate_check.json').exists())
        for text in ['no stamps\n','#8\n0a\n','#8\n#7\n']:
            p=self.root/'bad.vcd';p.write_text(text)
            with self.assertRaises(ValueError):normalizer.normalize(p)
            self.assertEqual(p.read_text(),text)


if __name__=='__main__':unittest.main()
