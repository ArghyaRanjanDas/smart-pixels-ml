"""Generated fixtures for constant ROM and named FSM reset lowering."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    helper = Path(__file__).with_name('sanitize_vitis_rtl.py')
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root/'image.dat').write_text('9\nf\n2\n')
        rtl = '''module fixture(input clk, input ap_rst, input [1:0] a, output [3:0] y);
parameter AddressRange = 3;
parameter IDLE = 1'd1;
reg state;
reg [3:0] rom0[0:AddressRange-1];
initial begin
#0 state = 1'd1;
end
initial begin
$readmemh("image.dat", rom0);
end
always @(posedge clk) begin
if (ap_rst) state <= IDLE;
else state <= 1'b0;
end
assign y = rom0[a];
endmodule
'''
        (root/'fixture.v').write_text(rtl)
        (root/'MANIFEST.json').write_text(json.dumps({'files': ['fixture.v', 'image.dat']}))
        out = root/'out.txt'
        result = subprocess.run([sys.executable, str(helper), str(root), str(out)], capture_output=True, text=True, check=True)
        report = json.loads(result.stdout)
        assert not report['unchecked'] and report['rom_arrays'] == 1
        text = out.read_text()
        assert 'initial begin' not in text and '$readmemh' not in text
        assert "assign rom0[1] = 'hf;" in text and 'wire [3:0] rom0' in text
        (root/'image.dat').write_text('9\nf\n')
        failed = subprocess.run([sys.executable, str(helper), str(root), str(out)], capture_output=True)
        assert failed.returncode != 0
    print('ROM lowering, alias reset, JSON-only output and partial-image rejection passed')


if __name__ == '__main__':
    main()
