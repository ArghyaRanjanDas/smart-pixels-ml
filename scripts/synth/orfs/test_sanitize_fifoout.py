"""Narrow FRP initializer acceptance and fail-closed regression fixtures."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

FIXTURE="""module fixture_frp_fifoout(input ap_clk, input ap_rst, input enabled, output value);
reg signed [2:0] fifo_rdPtr;
reg fifo_empty, fifo_full;
reg fail;
integer WarningSuppressThreshold;
initial begin // for co-sim assertion
    fifo_rdPtr = -1;
    fifo_empty = 1'b1;
    fifo_full = 1'b0;
    fail = 0;
    if(!$value$plusargs("warning_suppress_threshold=%0d", WarningSuppressThreshold))
        WarningSuppressThreshold = 5;
end
always @ (posedge ap_clk) begin : fifo_ptr
    if (ap_rst) begin
        fifo_rdPtr <= -1;
        fifo_empty <= 1'b1;
        fifo_full <= 1'b0;
    end
    else if (enabled) fifo_rdPtr <= fifo_rdPtr + 1;
end
// Simulation-only warning text may mention fail and WarningSuppressThreshold.
assign value = fifo_empty;
endmodule
"""


PIPELINE_FIXTURE="""module fixture_frp_pipeline_valid(input ap_clk, input ap_rst, input data, output value);
parameter PipelineLatency = 4;
reg fail;
integer WarningSuppressThreshold;
reg [PipelineLatency-2:0] v2;
reg [2:0] nvd_reg;
initial begin // for co-sim assertion
  v2 = 0;
  fail = 0;
  nvd_reg = 0;
  if(!$value$plusargs("warning_suppress_threshold=%0d", WarningSuppressThreshold))
    WarningSuppressThreshold = 5;
end
genvar i;
generate for (i = 0; i < PipelineLatency-1; i = i+1)
  always @ (posedge ap_clk) begin
    if (ap_rst)
      v2[i] <= 0;
    else v2[i] <= data;
  end
endgenerate
always @ (posedge ap_clk) begin
  if (ap_rst)
    nvd_reg <= 0;
  else nvd_reg <= nvd_reg + 1;
end
assign value = |v2;
endmodule
"""


class FIFOTests(unittest.TestCase):
    def sanitize(self,text,extra=None):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'fixture.v').write_text(text);out=root/'out.v'
            if extra is not None:(root/'top.v').write_text(extra)
            result=subprocess.run([sys.executable,str(Path(__file__).with_name('sanitize_vitis_rtl.py')),str(root),str(out)],capture_output=True,text=True)
            return result,out.read_text() if out.exists() else None

    def test_only_reset_equivalent_block_removed(self):
        result,text=self.sanitize(FIXTURE)
        self.assertEqual(result.returncode,0,result.stderr)
        report=json.loads(result.stdout);self.assertEqual(report['fifoout_reset_blocks'],1)
        self.assertEqual(report['initial_regs'],3);self.assertEqual(report['unchecked'],[])
        self.assertNotIn('initial begin',text);self.assertNotIn('$value$plusargs',text)
        # Synchronous reset and the functional next-state/output expressions stay intact.
        self.assertIn(FIXTURE[FIXTURE.index('always @'):],text)

    def test_complete_valid_vector_reset(self):
        result,text=self.sanitize(PIPELINE_FIXTURE)
        self.assertEqual(result.returncode,0,result.stderr)
        report=json.loads(result.stdout)
        self.assertEqual(report['pipeline_valid_reset_blocks'],1)
        self.assertEqual(report['initial_regs'],2)
        self.assertEqual(report['unchecked'],[])
        self.assertNotIn('initial begin',text)
        self.assertIn(PIPELINE_FIXTURE[PIPELINE_FIXTURE.index('genvar i;'):],text)

    def test_invalid_valid_vector_patterns_rejected(self):
        invalid=[
            PIPELINE_FIXTURE.replace('i = 0;', 'i = 1;'),
            PIPELINE_FIXTURE.replace('i < PipelineLatency-1;', 'i < PipelineLatency-2;'),
            PIPELINE_FIXTURE.replace('v2[i] <= 0;', 'v2[i] <= 1;'),
            PIPELINE_FIXTURE.replace('nvd_reg <= 0;', 'nvd_reg <= 1;'),
            PIPELINE_FIXTURE.replace('if (ap_rst)', 'if (data)'),
            PIPELINE_FIXTURE.replace('[PipelineLatency-2:0] v2', '[PipelineLatency-1:0] v2'),
            PIPELINE_FIXTURE.replace('v2 = 0;', 'v2 = 1;'),
            PIPELINE_FIXTURE.replace('assign value = |v2;', 'assign value = fail;'),
            PIPELINE_FIXTURE.replace('assign value = |v2;', 'assign value = WarningSuppressThreshold;'),
            PIPELINE_FIXTURE.replace('PipelineLatency = 4;', 'PipelineLatency = 1;'),
            PIPELINE_FIXTURE.replace('fail = 0;', 'fail = 0; nvd_reg = data;'),
        ]
        for text in invalid:
            result,out=self.sanitize(text)
            self.assertNotEqual(result.returncode,0)
            self.assertIsNone(out)
        for value in ('1','unknown_latency'):
            top='module top; fixture_frp_pipeline_valid #(.PipelineLatency('+value+')) instance0(); endmodule'
            result,out=self.sanitize(PIPELINE_FIXTURE,extra=top)
            self.assertNotEqual(result.returncode,0)
            self.assertIsNone(out)

    def test_invalid_patterns_still_rejected(self):
        invalid={
            'missing_reset':FIXTURE.replace('if (ap_rst)','if (enabled)'),
            'wrong_reset':FIXTURE.replace('fifo_rdPtr <= -1;','fifo_rdPtr <= 0;'),
            'wrong_initial':FIXTURE.replace('fifo_rdPtr = -1;','fifo_rdPtr = 0;'),
            'live_fail':FIXTURE.replace('assign value = fifo_empty;','assign value = fail;'),
            'live_warning':FIXTURE.replace('assign value = fifo_empty;','assign value = WarningSuppressThreshold;'),
            'extra_statement':FIXTURE.replace('fail = 0;','fail = 0; fifo_full = enabled;'),
            'other_plusarg':FIXTURE.replace('warning_suppress_threshold=%0d','other=%0d'),
            'duplicate_reset':FIXTURE.replace('fifo_rdPtr <= -1;','fifo_rdPtr <= -1; fifo_rdPtr <= 0;'),
            'non_fifo_module':FIXTURE.replace('fixture_frp_fifoout','unrelated_module'),
            'extra_initial':FIXTURE.replace('endmodule','initial begin fifo_full = enabled; end\nendmodule'),
        }
        for label,text in invalid.items():
            with self.subTest(label=label):
                result,output=self.sanitize(text)
                self.assertNotEqual(result.returncode,0)
                self.assertIsNone(output)


if __name__=='__main__':unittest.main()
