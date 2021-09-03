```python
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

```

# 目的
上面的code目的是为了重新测试在搜索空间中搜索出来的配置参数运行结果, 但是通过`lib = relay.build_module.build(mod, target=target, params=params)`时会报错, 如下所示, 因此想知道有什么其他的API用来测试我每次搜索空间搜索到的最优参数配置么?


```shell
(cg) zjq@DESKTOP-B5SJ9B4:tvm$ /home/zjq/miniconda3/envs/cg/bin/python /mnt/e/00_tvm/tvm/example/test.py
Compile...
Traceback (most recent call last):
  File "/mnt/e/00_tvm/tvm/example/test.py", line 64, in <module>
    test_tvm("random", "squeezenet_v1.1", 10)
  File "/mnt/e/00_tvm/tvm/example/test.py", line 40, in test_tvm
    lib = relay.build_module.build(mod, target=target, params=params)
  File "/mnt/e/00_tvm/tvm/python/tvm/relay/build_module.py", line 332, in build
    executor_config, runtime_mod, params = bld_mod.build(
  File "/mnt/e/00_tvm/tvm/python/tvm/relay/build_module.py", line 148, in build
    self._build(mod, target, target_host, executor)
  File "/mnt/e/00_tvm/tvm/python/tvm/_ffi/_ctypes/packed_func.py", line 237, in __call__
    raise get_last_ffi_error()
tvm._ffi.base.TVMError: Traceback (most recent call last):
  53: TVMFuncCall
  52: _ZNSt17_Function_handlerIFvN3tvm7runtime7TVMArgsEPNS1_11
  51: tvm::relay::backend::RelayBuildModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#3}::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
  50: tvm::relay::backend::RelayBuildModule::Build(tvm::IRModule, tvm::runtime::Map<tvm::Integer, tvm::Target, void, void> const&, tvm::Target const&, tvm::runtime::String)
  49: tvm::relay::backend::RelayBuildModule::BuildRelay(tvm::IRModule, std::unordered_map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, tvm::runtime::NDArray, std::hash<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::equal_to<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, tvm::runtime::NDArray> > > const&)
  48: tvm::relay::backend::ExecutorCodegen::Codegen(tvm::relay::Function const&)
  47: void tvm::relay::backend::ExecutorCodegen::CallFunc<tvm::relay::Function>(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::relay::Function)
  46: _ZNSt17_Function_handlerIFvN3tvm7runtime7TVMArgsEPNS1_11
  45: tvm::relay::backend::GraphExecutorCodegenModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#2}::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
  44: tvm::relay::backend::GraphExecutorCodegen::Codegen(tvm::relay::Function)
  43: tvm::relay::backend::MemoizedExprTranslator<std::vector<tvm::relay::backend::GraphNodeRef, std::allocator<tvm::relay::backend::GraphNodeRef> > >::VisitExpr(tvm::RelayExpr const&)
  42: tvm::relay::ExprFunctor<std::vector<tvm::relay::backend::GraphNodeRef, std::allocator<tvm::relay::backend::GraphNodeRef> > (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
  41: tvm::NodeFunctor<std::vector<tvm::relay::backend::GraphNodeRef, std::allocator<tvm::relay::backend::GraphNodeRef> > (tvm::runtime::ObjectRef const&, tvm::relay::ExprFunctor<std::vector<tvm::relay::backend::GraphNodeRef, std::allocator<tvm::relay::backend::GraphNodeRef> > (tvm::RelayExpr const&)>*)>::operator()(tvm::runtime::ObjectRef const&, tvm::relay::ExprFunctor<std::vector<tvm::relay::backend::GraphNodeRef, std::allocator<tvm::relay::backend::GraphNodeRef> > (tvm::RelayExpr const&)>*) const
  40: _ZZN3tvm5relay11ExprFunc
  39: tvm::relay::backend::GraphExecutorCodegen::VisitExpr_(tvm::relay::CallNode const*)
  38: tvm::relay::backend::GraphExecutorCodegen::GraphAddCallNode(tvm::relay::CallNode const*, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::unordered_map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, dmlc::any, std::hash<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::equal_to<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, dmlc::any> > >)
  37: tvm::relay::backend::MemoizedExprTranslator<std::vector<tvm::relay::backend::GraphNodeRef, std::allocator<tvm::relay::backend::GraphNodeRef> > >::VisitExpr(tvm::RelayExpr const&)
  36: tvm::relay::ExprFunctor<std::vector<tvm::relay::backend::GraphNodeRef, std::allocator<tvm::relay::backend::GraphNodeRef> > (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
  35: tvm::NodeFunctor<std::vector<tvm::relay::backend::GraphNodeRef, std::allocator<tvm::relay::backend::GraphNodeRef> > (tvm::runtime::ObjectRef const&, tvm::relay::ExprFunctor<std::vector<tvm::relay::backend::GraphNodeRef, std::allocator<tvm::relay::backend::GraphNodeRef> > (tvm::RelayExpr const&)>*)>::operator()(tvm::runtime::ObjectRef const&, tvm::relay::ExprFunctor<std::vector<tvm::relay::backend::GraphNodeRef, std::allocator<tvm::relay::backend::GraphNodeRef> > (tvm::RelayExpr const&)>*) const
  34: _ZZN3tvm5relay11ExprFunc
  33: tvm::relay::backend::GraphExecutorCodegen::VisitExpr_(tvm::relay::CallNode const*)
  32: tvm::relay::backend::GraphExecutorCodegen::GraphAddCallNode(tvm::relay::CallNode const*, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::unordered_map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, dmlc::any, std::hash<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::equal_to<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, dmlc::any> > >)
  31: tvm::relay::backend::MemoizedExprTranslator<std::vector<tvm::relay::backend::GraphNodeRef, std::allocator<tvm::relay::backend::GraphNodeRef> > >::VisitExpr(tvm::RelayExpr const&)
  30: tvm::relay::ExprFunctor<std::vector<tvm::relay::backend::GraphNodeRef, std::allocator<tvm::relay::backend::GraphNodeRef> > (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
  29: tvm::NodeFunctor<std::vector<tvm::relay::backend::GraphNodeRef, std::allocator<tvm::relay::backend::GraphNodeRef> > (tvm::runtime::ObjectRef const&, tvm::relay::ExprFunctor<std::vector<tvm::relay::backend::GraphNodeRef, std::allocator<tvm::relay::backend::GraphNodeRef> > (tvm::RelayExpr const&)>*)>::operator()(tvm::runtime::ObjectRef const&, tvm::relay::ExprFunctor<std::vector<tvm::relay::backend::GraphNodeRef, std::allocator<tvm::relay::backend::GraphNodeRef> > (tvm::RelayExpr const&)>*) const
  28: _ZZN3tvm5relay11ExprFunc
  27: tvm::relay::backend::GraphExecutorCodegen::VisitExpr_(tvm::relay::CallNode const*)
  26: tvm::relay::backend::GraphExecutorCodegen::GraphAddCallNode(tvm::relay::CallNode const*, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::unordered_map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, dmlc::any, std::hash<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::equal_to<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, dmlc::any> > >)
  25: tvm::relay::backend::MemoizedExprTranslator<std::vector<tvm::relay::backend::GraphNodeRef, std::allocator<tvm::relay::backend::GraphNodeRef> > >::VisitExpr(tvm::RelayExpr const&)
  24: tvm::relay::ExprFunctor<std::vector<tvm::relay::backend::GraphNodeRef, std::allocator<tvm::relay::backend::GraphNodeRef> > (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
  23: tvm::NodeFunctor<std::vector<tvm::relay::backend::GraphNodeRef, std::allocator<tvm::relay::backend::GraphNodeRef> > (tvm::runtime::ObjectRef const&, tvm::relay::ExprFunctor<std::vector<tvm::relay::backend::GraphNodeRef, std::allocator<tvm::relay::backend::GraphNodeRef> > (tvm::RelayExpr const&)>*)>::operator()(tvm::runtime::ObjectRef const&, tvm::relay::ExprFunctor<std::vector<tvm::relay::backend::GraphNodeRef, std::allocator<tvm::relay::backend::GraphNodeRef> > (tvm::RelayExpr const&)>*) const
  22: _ZZN3tvm5relay11ExprFunc
  21: tvm::relay::backend::GraphExecutorCodegen::VisitExpr_(tvm::relay::CallNode const*)
  20: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::relay::CachedFunc (tvm::relay::CompileEngine, tvm::relay::CCacheKey)>::AssignTypedLambda<tvm::relay::$_8>(tvm::relay::$_8, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
  19: tvm::relay::CompileEngineImpl::Lower(tvm::relay::CCacheKey const&)
  18: tvm::relay::CompileEngineImpl::LowerInternal(tvm::relay::CCacheKey const&)
  17: tvm::relay::CreateSchedule(tvm::relay::Function const&, tvm::Target const&)
  16: tvm::relay::ScheduleGetter::Create(tvm::relay::Function const&)
  15: tvm::relay::backend::MemoizedExprTranslator<tvm::runtime::Array<tvm::te::Tensor, void> >::VisitExpr(tvm::RelayExpr const&)
  14: tvm::relay::ExprFunctor<tvm::runtime::Array<tvm::te::Tensor, void> (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
  13: tvm::NodeFunctor<tvm::runtime::Array<tvm::te::Tensor, void> (tvm::runtime::ObjectRef const&, tvm::relay::ExprFunctor<tvm::runtime::Array<tvm::te::Tensor, void> (tvm::RelayExpr const&)>*)>::operator()(tvm::runtime::ObjectRef const&, tvm::relay::ExprFunctor<tvm::runtime::Array<tvm::te::Tensor, void> (tvm::RelayExpr const&)>*) const
  12: _ZZN3tvm5relay11ExprFunc
  11: tvm::relay::ScheduleGetter::VisitExpr_(tvm::relay::CallNode const*)
  10: tvm::relay::backend::MemoizedExprTranslator<tvm::runtime::Array<tvm::te::Tensor, void> >::VisitExpr(tvm::RelayExpr const&)
  9: tvm::relay::ExprFunctor<tvm::runtime::Array<tvm::te::Tensor, void> (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
  8: tvm::NodeFunctor<tvm::runtime::Array<tvm::te::Tensor, void> (tvm::runtime::ObjectRef const&, tvm::relay::ExprFunctor<tvm::runtime::Array<tvm::te::Tensor, void> (tvm::RelayExpr const&)>*)>::operator()(tvm::runtime::ObjectRef const&, tvm::relay::ExprFunctor<tvm::runtime::Array<tvm::te::Tensor, void> (tvm::RelayExpr const&)>*) const
  7: _ZZN3tvm5relay11ExprFunc
  6: tvm::relay::ScheduleGetter::VisitExpr_(tvm::relay::CallNode const*)
  5: tvm::relay::backend::MemoizedExprTranslator<tvm::runtime::Array<tvm::te::Tensor, void> >::VisitExpr(tvm::RelayExpr const&)
  4: tvm::relay::ExprFunctor<tvm::runtime::Array<tvm::te::Tensor, void> (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
  3: tvm::NodeFunctor<tvm::runtime::Array<tvm::te::Tensor, void> (tvm::runtime::ObjectRef const&, tvm::relay::ExprFunctor<tvm::runtime::Array<tvm::te::Tensor, void> (tvm::RelayExpr const&)>*)>::operator()(tvm::runtime::ObjectRef const&, tvm::relay::ExprFunctor<tvm::runtime::Array<tvm::te::Tensor, void> (tvm::RelayExpr const&)>*) const
  2: _ZZN3tvm5relay11ExprFunc
  1: tvm::relay::ScheduleGetter::VisitExpr_(tvm::relay::CallNode const*)
  0: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), TVMFuncCreateFromCFunc::$_2>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
  File "/mnt/e/00_tvm/tvm/python/tvm/_ffi/_ctypes/packed_func.py", line 81, in cfun
    rv = local_pyfunc(*pyargs)
  File "/mnt/e/00_tvm/tvm/python/tvm/relay/backend/compile_engine.py", line 311, in lower_call
    best_impl, outputs = select_implementation(op, call.attrs, inputs, ret_type, target)
  File "/mnt/e/00_tvm/tvm/python/tvm/relay/backend/compile_engine.py", line 219, in select_implementation
    outs = impl.compute(attrs, inputs, out_type)
  File "/mnt/e/00_tvm/tvm/python/tvm/relay/op/op.py", line 90, in compute
    return _OpImplementationCompute(self, attrs, inputs, out_type)
  File "/mnt/e/00_tvm/tvm/python/tvm/_ffi/_ctypes/packed_func.py", line 237, in __call__
    raise get_last_ffi_error()
  3: TVMFuncCall
  2: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::relay::$_3>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
  1: tvm::relay::OpImplementation::Compute(tvm::Attrs const&, tvm::runtime::Array<tvm::te::Tensor, void> const&, tvm::Type const&)
  0: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), TVMFuncCreateFromCFunc::$_2>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
  File "/mnt/e/00_tvm/tvm/python/tvm/_ffi/_ctypes/packed_func.py", line 81, in cfun
    rv = local_pyfunc(*pyargs)
  File "/mnt/e/00_tvm/tvm/python/tvm/relay/op/strategy/generic.py", line 240, in _compute_conv2d
    return [topi_compute(*args)]
  File "/mnt/e/00_tvm/tvm/python/tvm/autotvm/task/topi_integration.py", line 161, in wrapper
    cfg = DispatchContext.current.query(tgt, workload)
  File "/mnt/e/00_tvm/tvm/python/tvm/autotvm/task/dispatcher.py", line 76, in query
    ret = self._query_inside(target, workload)
  File "/mnt/e/00_tvm/tvm/python/tvm/autotvm/task/dispatcher.py", line 421, in _query_inside
    assert wkl == workload
TVMError: AssertionError
```