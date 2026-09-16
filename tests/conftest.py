"""测试全局配置（pytest 自动加载，早于任何测试模块 import）。

★ 这里必须在**任何 app 模块被 import 之前**把 OTEL 关掉：
  app.main 在 import 时就会装配 TracerProvider + FastAPI 自动埋点（追踪必须赶在应用
  启动前装好，见 tracing.setup_tracing），而测试不该连真实 Jaeger —— 起容器会卡住，
  4317 不通则让后台导出线程刷一堆错误日志，断言还会被这种噪音淹没。

  追踪本身的用例自己在内存里装 exporter（tests/test_tracing.py 的 module fixture），
  所以关掉全局开关不影响它们。
"""

import os

# 直接赋值而非 setdefault：测试必须**确定性**地不依赖外部 Jaeger，
# shell 里恰好 export 了 OTEL_ENABLED=true 也不能把测试带跑偏。
# （pydantic-settings 的优先级是 环境变量 > .env 文件，所以这行也能盖住用户的 .env）
os.environ["OTEL_ENABLED"] = "false"
