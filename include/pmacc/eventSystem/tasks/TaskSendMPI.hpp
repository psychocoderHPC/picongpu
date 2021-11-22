/* Copyright 2013-2021 Felix Schmitt, Rene Widera, Wolfgang Hoenig,
 *                     Benjamin Worpitz
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

#include "pmacc/Environment.hpp"
#include "pmacc/communication/ICommunicator.hpp"
#include "pmacc/eventSystem/tasks/MPITask.hpp"
#include "pmacc/memory/buffers/Buffer.hpp"

#include <mpi.h>

namespace pmacc
{
    template<class TYPE, unsigned DIM>
    class Exchange;

    template<class TYPE, unsigned DIM>
    class TaskSendMPI : public MPITask
    {
    public:
        TaskSendMPI(Exchange<TYPE, DIM>* exchange) : MPITask(), exchange(exchange)
        {
        }

        void init() override
        {
            if(Environment<>::get().isMpiDirectEnabled())
            {
                /* Wait to be sure that all device work is finished before MPI is triggered.
                 * MPI will not wait for work in our device streams
                 */
                mpiDirectInitDependency = __getTransactionEvent();
                state = WaitForInitDependency;
            }
            else
            {
                startSend();
            }
        }


        bool executeIntern() override
        {
            switch(state)
            {
            case WaitForInitDependency:
                if(nullptr
                   == Environment<>::get().Manager().getITaskIfNotFinished(mpiDirectInitDependency.getTaskId()))
                {
                    state = UpdateState;
                    // do not block the event system, dependencies already covered by mpiDirectInitDependency
                    __startTransaction();
                    startSend();
                    __endTransaction();
                    state = WaitForMpiOperation;
                }
                break;
            case WaitForMpiOperation:
                {
                    if(this->request == nullptr)
                        throw std::runtime_error("request was nullptr (call executeIntern after freed");

                    int flag = 0;
                    MPI_CHECK(MPI_Test(this->request, &flag, &(this->status)));

                    if(flag) // finished
                    {
                        state = Finish;
                        delete this->request;
                        this->request = nullptr;
                        return true;
                    }
                }
                break;
            case Finish:
                return true;
            default:
                return false;
            }
            return false;
        }

        ~TaskSendMPI() override
        {
            notify(this->myId, SENDFINISHED, nullptr);
        }

        void event(id_t, EventType, IEventData*) override
        {
        }

        std::string toString() override
        {
            return std::string("TaskSendMPI exchange type=") + std::to_string(exchange->getExchangeType());
        }

    private:
        /** Send data via MPI
         *
         * @attention This operation could be blocking because of the access to exchange and src.
         *            Take care if you call this method from executeIntern.
         */
        void startSend()
        {
            Buffer<TYPE, DIM>* src = exchange->getCommunicationBuffer();

            this->request = Environment<DIM>::get().EnvironmentController().getCommunicator().startSend(
                exchange->getExchangeType(),
                reinterpret_cast<char*>(src->getPointer()),
                src->getCurrentSize() * sizeof(TYPE),
                exchange->getCommunicationTag());
            state = WaitForMpiOperation;
        }

        enum state_t
        {
            WaitForInitDependency,
            WaitForMpiOperation,
            UpdateState,
            Finish
        };

        Exchange<TYPE, DIM>* exchange;
        MPI_Request* request;
        MPI_Status status;
        EventTask mpiDirectInitDependency;
        state_t state = UpdateState;
    };

} // namespace pmacc
