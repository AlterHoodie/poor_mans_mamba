#pragma once

// Minimal leveled logger. Level comes from env MAMBASERVE_LOG
// (debug | info | warn | error | off), default "warn".
//
// Rules of use:
//  - Above debug level, never log while holding session_mu_ / event_mu_ (format + I/O is
//    slow). Debug logging in control paths is tolerated because timed runs keep it off.
//  - Per-token logging belongs at LOG_DEBUG only; benchmark runs use warn/off.
//  - Every line should carry req_id= and worker= where applicable so log lines
//    can be joined with the trace CSV.

#include <atomic>

namespace telemetry {

enum class LogLevel : int { Debug = 0, Info = 1, Warn = 2, Error = 3, Off = 4 };

// Constant-initialised; overridden from the environment during static init
// (see log.cpp) so the hot-path check below is a single relaxed load.
extern std::atomic<int> g_log_level;

inline bool log_enabled(LogLevel lvl) {
  return static_cast<int>(lvl) >= g_log_level.load(std::memory_order_relaxed);
}

inline LogLevel log_level() {
  return static_cast<LogLevel>(g_log_level.load(std::memory_order_relaxed));
}

void set_log_level(LogLevel lvl);
LogLevel parse_log_level(const char* s, LogLevel fallback);

// Writes "<ms-since-start> <LEVEL> <message>\n" to stderr (single fprintf call).
void log_write(LogLevel lvl, const char* fmt, ...) __attribute__((format(printf, 2, 3)));

} // namespace telemetry

#define MS_LOG(lvl, ...)                                                                           \
  do {                                                                                             \
    if (::telemetry::log_enabled(lvl))                                                             \
      ::telemetry::log_write(lvl, __VA_ARGS__);                                                    \
  } while (0)

#define LOG_DEBUG(...) MS_LOG(::telemetry::LogLevel::Debug, __VA_ARGS__)
#define LOG_INFO(...) MS_LOG(::telemetry::LogLevel::Info, __VA_ARGS__)
#define LOG_WARN(...) MS_LOG(::telemetry::LogLevel::Warn, __VA_ARGS__)
#define LOG_ERROR(...) MS_LOG(::telemetry::LogLevel::Error, __VA_ARGS__)
