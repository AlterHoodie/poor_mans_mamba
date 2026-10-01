#include "telemetry/log.h"

#include "telemetry/recorder.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <strings.h>

namespace telemetry {

std::atomic<int> g_log_level{static_cast<int>(LogLevel::Warn)};

LogLevel parse_log_level(const char* s, LogLevel fallback) {
  if (!s || !*s)
    return fallback;
  if (!strcasecmp(s, "debug"))
    return LogLevel::Debug;
  if (!strcasecmp(s, "info"))
    return LogLevel::Info;
  if (!strcasecmp(s, "warn") || !strcasecmp(s, "warning"))
    return LogLevel::Warn;
  if (!strcasecmp(s, "error"))
    return LogLevel::Error;
  if (!strcasecmp(s, "off") || !strcasecmp(s, "none"))
    return LogLevel::Off;
  return fallback;
}

void set_log_level(LogLevel lvl) { g_log_level.store(static_cast<int>(lvl), std::memory_order_relaxed); }

namespace {

const int64_t g_start_ns = steady_now_ns();

struct EnvInit {
  EnvInit() { set_log_level(parse_log_level(std::getenv("MAMBASERVE_LOG"), LogLevel::Warn)); }
} g_env_init;

const char* level_name(LogLevel l) {
  switch (l) {
  case LogLevel::Debug:
    return "DEBUG";
  case LogLevel::Info:
    return "INFO ";
  case LogLevel::Warn:
    return "WARN ";
  case LogLevel::Error:
    return "ERROR";
  case LogLevel::Off:
    break;
  }
  return "?";
}

} // namespace

void log_write(LogLevel lvl, const char* fmt, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  const double ms = static_cast<double>(steady_now_ns() - g_start_ns) / 1e6;
  // One fprintf call => stdio's per-FILE lock keeps lines from interleaving.
  fprintf(stderr, "%10.3f %s %s\n", ms, level_name(lvl), buf);
}

} // namespace telemetry
