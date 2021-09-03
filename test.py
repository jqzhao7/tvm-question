# Load packages
import numpy as np
import os
import tvm
from tvm import relay, autotvm
from tvm.relay import testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
# import tvm.contrib.graph_runtime as runtime
import tvm.contrib.graph_executor as runtime
import os
import time
target = "llvm"
dtype = "float32"
batch_size = 1

def get_network(name, batch_size):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)
    mod, params = relay.testing.squeezenet.get_workload(
        batch_size=batch_size, version="1.1", dtype=dtype
    )
    return mod, params, input_shape, output_shape

def test_tvm(tuner_name, model_name, n_t=0):
    mod, params, data_shape, out_shape = get_network(model_name, batch_size)
    now_time = time.strftime("now time : %Y-%m-%d %H:%M:%S", time.localtime())
    start_time = time.perf_counter()
    graph_opt_sch_file = f"example/{model_name}-{tuner_name}-{n_t}_graph_opt.log"
    num_threads = 1
    os.environ["TVM_NUM_THREADS"] = str(num_threads)
    input_name = "data"

    start = time.perf_counter()
    # compile kernels with graph-level best records
    with autotvm.apply_graph_best(graph_opt_sch_file):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build_module.build(mod, target=target, params=params)

        # upload parameters to device
        dev = tvm.cpu()
        data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype(dtype))
        module = runtime.GraphModule(lib["default"](dev))
        module.set_input(input_name, data_tvm)

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", dev, number=100, repeat=3)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print(
            f"{model_name}-{tuner_name}: Mean inference time (std dev): %.2f ms (%.2f ms) geomean=%.2f"
            % (np.mean(prof_res), np.std(prof_res), np.exp(np.mean(np.log(prof_res))))
        )
        with open("result.log", "a") as f:
            f.write(f"{now_time} -- {model_name} -- {tuner_name} -- {n_t} -- Mean=%.2f ms (%.2f ms) geomean=%.2f"%(
                np.mean(prof_res), 
                np.std(prof_res), 
                np.exp(np.mean(np.log(prof_res)))))
            end_time = time.perf_counter()
            print("time cost(s):  %0.2f s\n\n\n"%float(end_time-start_time))
            f.write("  timeCost=%0.2fs\n"%float(end_time-start_time))
test_tvm("random", "squeezenet_v1.1", 10)