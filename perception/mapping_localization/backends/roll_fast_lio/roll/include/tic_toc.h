// Author:   Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk

#pragma once

#include <ctime>
#include <cstdlib>
#include <chrono>
#include <string>
#include <iostream>

class TicToc
{
  public:
    TicToc()
    {
        tic();
    }

    void tic()
    {
        start = std::chrono::system_clock::now();
    }

    double toc()
    {
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start; // micro secs or us
        return elapsed_seconds.count() * 1000; //ms
    }

    void toc(std::string _about_task) {
      end = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed_seconds = end - start;
      double elapsed_ms = elapsed_seconds.count() * 1000;

      std::cout.precision(3);  // 10 for sec, 3 for ms
      std::cout << _about_task << ": " << elapsed_ms << " msec." << std::endl;
    }

  private:
    std::chrono::time_point<std::chrono::system_clock> start, end;
};
