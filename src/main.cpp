#include "core/device.h"
#include "runtime/cache/cache_layout.h"
#include "runtime/cache/cache_pool.h"
#include "runtime/model_registry.h"
#include "runtime/tokenizer.h"

#include <iostream>
#include <memory>
#include <string>
#include <string_view>

int main(int argc, char** argv) {
  const std::string model_dir = (argc > 1) ? argv[1] : "models/mamba2-130m-hf";
  const int max_seq_length = (argc > 2) ? std::stoi(argv[2]) : 2048;
  const Device device =
      (argc > 3 && std::string_view(argv[3]) == "GPU") ? Device::GPU : Device::CPU;

  auto entry_or = ModelRegistry::open(model_dir, max_seq_length);
  if (!entry_or.ok()) {
    std::cerr << entry_or.status().message() << '\n';
    return 1;
  }
  ModelEntry entry = std::move(entry_or.value());
  const ModelConfig& cfg = *entry.cfg;

  std::unique_ptr<DeviceAllocator> alloc;
  auto alloc_or = create_device_allocator(device, /*device_id=*/0);
  if (!alloc_or.ok()) {
    std::cerr << alloc_or.status().message() << '\n';
    return 1;
  }
  alloc = std::move(alloc_or.value());

  auto runner_or = entry.create_runner(*alloc);
  if (!runner_or.ok()) {
    std::cerr << runner_or.status().message() << '\n';
    return 1;
  }
  std::unique_ptr<Runner> runner = std::move(runner_or.value());

  auto pool_or = create_cache_pool(cfg, alloc.get(), /*num_slots=*/10, create_cache_layout);
  if (!pool_or.ok()) {
    std::cerr << pool_or.status().message() << '\n';
    return 1;
  }
  std::unique_ptr<CachePool> pool = std::move(pool_or.value());

  std::cout << "model_type=" << cfg.model_type << '\n'
            << "layers=" << cfg.num_hidden_layers << '\n'
            << "hidden_size=" << cfg.hidden_size << '\n';

  Tokenizer tokenizer(cfg, model_dir);

  std::string prompt = "Hi How are you?";
  auto ids = tokenizer.encode(prompt);
  if (!ids.ok()) {
    std::cerr << ids.status().message() << '\n';
    return 1;
  }

  auto handle_or = pool->acquire();
  if (!handle_or.ok()) {
    std::cerr << handle_or.status().message() << '\n';
    return 1;
  }
  CacheHandle handle = std::move(handle_or.value());

  auto layers_or = pool->layer_views(handle);
  if (!layers_or.ok()) {
    std::cerr << layers_or.status().message() << '\n';
    return 1;
  }

  auto logits_or = [&]() -> StatusOr<Tensor> {
    AllocatorScope scope(alloc.get());
    return runner->prefill(ids.value(), layers_or.value());
  }();
  if (!logits_or.ok()) {
    std::cerr << logits_or.status().message() << '\n';
    return 1;
  }

  std::cout << "prefill ok; logits rank=" << logits_or.value().shape.size() << '\n';
  (void)pool->release(handle);
  return 0;
}
