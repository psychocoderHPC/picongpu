/* Copyright 2019 Rene Widera
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

namespace pmacc
{
    using Clock = std::chrono::high_resolution_clock;
    template<class Duration>
    using TimePoint = std::chrono::time_point<Clock, Duration>;
    using Milliseconds = std::chrono::milliseconds;

    class PerfData
    {
    private:
        using region_t = std::tuple<std::string, double, int64_t>;

    public:
        static PerfData& inst()
        {
            static PerfData instance;
            return instance;
        }

        double getTime()
        {
            auto time(Clock::now().time_since_epoch());
            auto timestamp = std::chrono::duration_cast<Milliseconds>(time).count();
            return static_cast<double>(timestamp);
        }

        void pushRegions(std::string name, double const& duration, int64_t const timeStep = -1)
        {
            regions.emplace_back(region_t(std::move(name), duration, timeStep));
        }

        void activate()
        {
            showReport = true;
        }

        ~PerfData()
        {
            if(showReport)
            {
                for(auto const& r : regions)
                {
                    std::cout << "report: " << std::get<0>(r) << " " << std::get<1>(r) << " " << std::get<2>(r)
                              << std::endl;
                }
            }
        }


    private:
        PerfData() = default;

        std::vector<region_t> regions;
        bool showReport = false;
    };

} // namespace pmacc
