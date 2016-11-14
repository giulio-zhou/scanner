#include "scanner/engine.h"
#include "scanner/evaluators/util/discard_evaluator.h"
#include "scanner/evaluators/video/decoder_evaluator.h"

namespace scanner {
namespace {
PipelineDescription get_pipeline_description(
    const DatasetMetadata& dataset_meta,
    const std::vector<DatasetItemMetadata>& item_metas) {
  PipelineDescription desc;
  desc.input_columns = {"frame"};

  std::vector<std::unique_ptr<EvaluatorFactory>>& factories =
      desc.evaluator_factories;

  factories.emplace_back(new DiscardEvaluatorFactory(DeviceType::CPU));

  return desc;
}
}

REGISTER_PIPELINE(effective_io_rate, get_pipeline_description);
}