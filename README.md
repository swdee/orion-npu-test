# Orion O6 NPU Test

I am wanting to gather the results of running a YOLOv8 model on the Orion O6 NPU.  Would you please assist me with this by following
the instructions below.

**NOTE: This process requires Radxa's Debian OS image.**


## Instructions

Use git to obtain test files and data.
```
cd /tmp
git clone https://github.com/swdee/orion-npu-test.git
```

Install python virtual environment.
```
sudo apt install python3.11-venv
```

Create virtual environment for project.
```
cd orion-npu-test
python3 -m venv --system-site-packages .venv
```

Install the needed python packages into the virtual environment.
```
.venv/bin/pip install -r utils/requirements.txt
```

Run the inference test script.
```
.venv/bin/python inference_npu.py 
```

The script will output something similar to the following.   Please copy this text
and add it to a post on this [forum thread](https://forum.radxa.com/t/cixbuilder-problems-compiling-onnx-model-slow-inference-times/26972) 
OR [open an issue](https://github.com/swdee/orion-npu-test/issues) on the Github project.

Thank you.
```
=== System info: Linux orion-o6 6.1.44-cix-build-generic #2 SMP PREEMPT Fri Jan 24 18:51:41 CST 2025 aarch64  ===

npu: noe_init_context success
npu: noe_load_graph success
Input tensor count is 1.
Output tensor count is 1.
npu: noe_create_job success

=== Detections for palace.jpg ===
  [0] class=0, conf=0.894, bbox=(179.8,224.7,339.6,696.6)
  [1] class=0, conf=0.505, bbox=(160.4,197.4,240.6,366.6)
  [2] class=0, conf=0.505, bbox=(260.7,231.2,361.0,344.0)
  [3] class=0, conf=0.505, bbox=(586.6,211.5,696.9,454.0)
  [4] class=0, conf=0.505, bbox=(386.0,239.7,516.4,527.3)
  [5] class=0, conf=0.505, bbox=(752.0,248.2,852.3,507.6)
  [6] class=0, conf=0.505, bbox=(651.7,273.5,772.1,504.8)
  [7] class=0, conf=0.505, bbox=(541.4,290.5,661.8,521.7)
  [8] class=0, conf=0.505, bbox=(315.8,338.4,406.1,575.3)
  [9] class=0, conf=0.311, bbox=(1198.2,335.6,1288.4,679.6)
  [10] class=0, conf=0.874, bbox=(948.8,224.7,1168.5,719.1)
  [11] class=0, conf=0.466, bbox=(1148.6,314.6,1328.3,561.8)
  [12] class=0, conf=0.525, bbox=(858.9,236.0,1657.9,820.2)
  [13] class=26, conf=0.505, bbox=(396.1,291.9,486.3,396.2)
  [14] class=26, conf=0.408, bbox=(799.6,286.2,874.8,384.9)
  [15] class=26, conf=0.408, bbox=(130.3,335.6,190.5,442.7)
  [16] class=26, conf=0.311, bbox=(674.3,396.2,719.4,483.6)
  [17] class=26, conf=0.311, bbox=(937.5,479.4,997.7,626.0)
  [18] class=26, conf=0.311, bbox=(1158.1,530.2,1248.3,676.8)

Inference over 10 runs:
  avg = 114.55 ms   min = 107.69 ms   max = 120.52 ms
npu: noe_clean_job success
npu: noe_unload_graph success
npu: noe_deinit_context success
```


## Clean Up

After running the test and posting your output to the forum or github, this project can be deleted from your system.
```
rm -rf orion-npu-test
```
