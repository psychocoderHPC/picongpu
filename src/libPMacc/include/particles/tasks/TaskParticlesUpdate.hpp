/**
 * Copyright 2013 Rene Widera
 *
 * This file is part of libPMacc. 
 * 
 * libPMacc is free software: you can redistribute it and/or modify 
 * it under the terms of of either the GNU General Public License or 
 * the GNU Lesser General Public License as published by 
 * the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version. 
 * libPMacc is distributed in the hope that it will be useful, 
 * but WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
 * GNU General Public License and the GNU Lesser General Public License 
 * for more details. 
 * 
 * You should have received a copy of the GNU General Public License 
 * and the GNU Lesser General Public License along with libPMacc. 
 * If not, see <http://www.gnu.org/licenses/>. 
 */

#pragma once


#include "Environment.hpp"
#include "eventSystem/EventSystem.hpp"


namespace PMacc
{

template<class ParBase>
class TaskParticlesUpdate : public MPITask
{
public:


    EventTask serialEvent;
    EventTask commEvent;
    uint32_t currentStep;

    TaskParticlesUpdate(ParBase &parBase, uint32_t currentStep,EventTask ev) :
    parBase(parBase),
    state(Constructor), currentStep(currentStep),serialEvent(ev)
    {
    }

    virtual void init()
    {
        state = Init;
        //serialEvent = __getTransactionEvent();


    }

    bool executeIntern()
    {
        switch (state)
        {
        case Init:
            if (serialEvent.isFinished())
                state = CallUpdate;
            break;
        case CallUpdate:
            state = Unknown;
            __startTransaction();
            parBase.update(currentStep);
            serialEvent = __endTransaction();
            state = WaitForUpdate;
            break;
        case WaitForUpdate:
            if (serialEvent.isFinished())
                state = CallShift;
            break;
        case CallShift:
            state = Unknown;
            __startTransaction();
            parBase.template shiftParticles < CORE + BORDER > ();
            serialEvent = __endTransaction();
            state = WaitForShift;
            break;
        case WaitForShift:
            if (serialEvent.isFinished())
                state = CallFillGapsBorder;
            break;
        case CallFillGapsBorder:
            state = Unknown;
            __startAtomicTransaction();
            parBase.template fillGaps < BORDER > ();
            parBase.template fillGaps < GUARD > ();
            serialEvent = __endTransaction();
            state = WaitForFillGapsBorder;
            break;
        case WaitForFillGapsBorder:
            if (serialEvent.isFinished())
                state = CallFillGapsCore;
            break;
        case CallFillGapsCore:
            state = Unknown;
            commEvent = parBase.asyncCommunication(EventTask());
            __startTransaction();
            parBase.template fillGaps < CORE > ();
            serialEvent = __endTransaction();
            state = WaitForFillGapsCore;
            break;
        case WaitForFillGapsCore:
            if (serialEvent.isFinished() && commEvent.isFinished())
                return true;
            break;
        default:
            return false;
        }

        return false;
    }

    virtual ~TaskParticlesUpdate()
    {
        notify(this->myId, RECVFINISHED, NULL);
    }

    void event(id_t, EventType, IEventData*)
    {
    }

    std::string toString()
    {
        return "TaskParticlesUpdate";
    }

private:

    enum state_t
    {
        Constructor,
        Init,
        Unknown,
        CallUpdate,
        WaitForUpdate,
        CallShift,
        WaitForShift,
        CallFillGapsBorder,
        WaitForFillGapsBorder,
        CallFillGapsCore,
        WaitForFillGapsCore

    };


    ParBase& parBase;
    state_t state;

};

} //namespace PMacc

