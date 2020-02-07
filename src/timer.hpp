/*
 * Copyright 2018- Yamana Laboratory, Waseda University
 */

#pragma once

#include <chrono>
#include <iomanip>
#include <string>

class Timer
{
public:
    using T = std::chrono::steady_clock;
    using Period = std::chrono::microseconds;
    inline void Start() noexcept;
    inline void Stop() noexcept;
    inline Timer::Period Duration();
    inline std::string ReturnCurrentTimeAndDate();

private:
    T::time_point _start_time;
    T::time_point _current_time;
    bool _is_timer_working = false;
};

inline void Timer::Start() noexcept
{
    _start_time = T::now();
    _current_time = T::now();
    _is_timer_working = true;
}

inline void Timer::Stop() noexcept
{
    _current_time = T::now();
    _is_timer_working = false;
}

// Return duration
inline Timer::Period Timer::Duration()
{
    if (_is_timer_working)
    {
        throw std::logic_error("Timer is working.");
    }
    return std::chrono::duration_cast<Period>(_current_time - _start_time);
}

inline std::string Timer::ReturnCurrentTimeAndDate()
{
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
    return ss.str();
}