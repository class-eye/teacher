#include <chrono>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>

#ifdef WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

class Timer {
	using Clock = std::chrono::high_resolution_clock;

public:
	/*! \brief start or restart timer */
	inline void Tic() {
		start_ = Clock::now();
	}
	/*! \brief stop timer */
	inline void Toc() {
		end_ = Clock::now();
	}
	/*! \brief return time in ms */
	inline double Elasped() {
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_ - start_);
		return duration.count();
	}
	const std::string getCurrentSystemTime()
	{
		auto tt = std::chrono::system_clock::to_time_t
			(std::chrono::system_clock::now());
		struct tm* ptm = localtime(&tt);
		char date[60] = { 0 };
		sprintf(date, "%d-%02d-%02d %02d:%02d:%02d",
			(int)ptm->tm_year + 1900, (int)ptm->tm_mon + 1, (int)ptm->tm_mday,
			(int)ptm->tm_hour, (int)ptm->tm_min, (int)ptm->tm_sec);
		return std::string(date);
	}
	string getFileCreateTime(string &file){
		struct stat buf;
		int result;
		result = stat(file.c_str(), &buf);

		if (result != 0)
			perror("No such file or directory");
		else
		{
			string str_time = ctime(&buf.st_mtime);
			char buff[300];
			sprintf(buff,"%s_%s_%s %s",str_time.substr(20,4).c_str(),str_time.substr(4,3).c_str(),str_time.substr(8,2).c_str(),str_time.substr(11,8).c_str());
			string create_time=buff;
			return create_time;

		}
	}
	void sleep_(int t){
#ifdef WIN32
		Sleep(t);
#else
		sleep(t);
#endif

	}
private:
	Clock::time_point start_, end_;
};