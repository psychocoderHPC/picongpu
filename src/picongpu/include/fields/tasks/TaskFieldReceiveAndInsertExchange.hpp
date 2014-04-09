/**
 * Copyright 2013 Rene Widera
 *
 * This file is part of PIConGPU. 
 * 
 * PIConGPU is free software: you can redistribute it and/or modify 
 * it under the terms of the GNU General Public License as published by 
 * the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version. 
 * 
 * PIConGPU is distributed in the hope that it will be useful, 
 * but WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
 * GNU General Public License for more details. 
 * 
 * You should have received a copy of the GNU General Public License 
 * along with PIConGPU.  
 * If not, see <http://www.gnu.org/licenses/>. 
 */ 
 


#ifndef _TASKFIELDRECEIVEANDINSERTEXCHANGE_HPP
#define	_TASKFIELDRECEIVEANDINSERTEXCHANGE_HPP


#include "eventSystem/EventSystem.hpp"
#include "fields/tasks/FieldFactory.hpp"
#include "eventSystem/tasks/ITask.hpp"
#include "eventSystem/tasks/MPITask.hpp"
#include "eventSystem/events/EventDataReceive.hpp"



namespace PMacc
{

template<class Field>
class TaskFieldReceiveAndInsertExchange : public MPITask
{
public:

    TaskFieldReceiveAndInsertExchange(Field &buffer, uint32_t exchange,EventTask ev) :
    buffer(buffer),
    exchange(exchange),
    state(Constructor),
    initDependency(ev)
    {
    }

    virtual void init()
    {
        state = PreInit;

    }

    bool executeIntern()
    {
        switch (state)
        {
        case PreInit:
            if (initDependency.isFinished())
                state = Init;
            break;
        case Init:
            state=InitWait;
            initDependency = buffer.getGridBuffer().asyncReceive(initDependency, exchange);
            state = WaitForReceive;
            break;
        case WaitForReceive:
            if (NULL == Environment<>::get().Manager().getITaskIfNotFinished(initDependency.getTaskId()))
            {
                state = Finished;
                return true;
            }
            break;
        case Finished:
            return true;
        default:
            return false;
        }

        return false;
    }

    virtual ~TaskFieldReceiveAndInsertExchange()
    {
        notify(this->myId, RECVFINISHED, NULL);
    }

    void event(id_t, EventType, IEventData*)
    {
    }

    std::string toString()
    {
        std::ostringstream stateNumber;
        stateNumber << state;
        return std::string("TaskFieldReceiveAndInsertExchange/") + stateNumber.str();
    }

private:

    enum state_t
    {
        Constructor,
        Init,
        WaitForReceive,
        Finished,
            InitWait,
            PreInit

    };




    Field& buffer;
    state_t state;
    EventTask insertEvent;
    EventTask initDependency;
    uint32_t exchange;
};

} //namespace PMacc


#endif	/* _TASKFIELDRECEIVEANDINSERTEXCHANGE_HPP */

