#include "teacher/Timer.hpp"
#include <chrono>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>

using Clock = std::chrono::high_resolution_clock;

void Timer::Tic() {
	time_record time1;
	time1.start_ = Clock::now();
	record.push_back(time1);
}
/*! \brief stop timer */
void Timer::Toc() {
	time_record time2 = record.back();
	time2.end_ = Clock::now();
	record_pair.push_back(time2);
	record.pop_back();
}
/*! \brief return time in ms */
double Timer::Elasped() {

	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(record_pair[0].end_-record_pair[0].start_);
	record_pair.clear();
	return duration.count();
}