/*
 * Copyright 2018- Yamana Laboratory, Waseda University
 */

#pragma once

#include <chrono>
#include <iomanip>
#include <string>

class Timer {
public:
  inline void Start() noexcept;
  inline void Stop() noexcept;
  template <class Rep> inline std::chrono::duration<Rep> Duration();
  inline std::string ReturnCurrentTimeAndDate();

private:
  std::chrono::system_clock::time_point _start_time;
  std::chrono::system_clock::time_point _current_time;
  bool _is_timer_working = false;
};

inline void Timer::Start() noexcept {
  _start_time = std::chrono::system_clock::now();
  _current_time = std::chrono::system_clock::now();
  _is_timer_working = true;
}

inline void Timer::Stop() noexcept {
  _current_time = std::chrono::system_clock::now();
  _is_timer_working = false;
}

template <class Rep> inline std::chrono::duration<Rep> Timer::Duration() {
  if (_is_timer_working) {
    throw std::logic_error("Timer is working.");
  }
  return _current_time - _start_time;
}

inline std::string Timer::ReturnCurrentTimeAndDate() {
  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);

  std::stringstream ss;
  ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
  return ss.str();
}