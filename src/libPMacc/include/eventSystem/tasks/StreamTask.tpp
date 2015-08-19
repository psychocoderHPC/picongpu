/**
 * Copyright 2013-2015 Rene Widera, Benjamin Worpitz
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
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


#include "Environment.hpp"
//#include "eventSystem/EventSystem.hpp"
#include "eventSystem/tasks/StreamTask.hpp"
#include "eventSystem/streams/EventStream.hpp"

#include <cassert>

namespace PMacc
{

inline StreamTask::StreamTask( ) :
ITask( ),
stream( NULL ),
cudaEvent( ),
alwaysFinished( false )
{
    this->setTaskType( TASK_CUDA );
}

inline CudaEvent StreamTask::getCudaEvent( ) const
{
    assert(cudaEvent);
    return *cudaEvent.get();
}

inline void StreamTask::setCudaEvent(const CudaEvent& cudaEvent )
{
    this->cudaEvent.reset(new CudaEvent(cudaEvent));
}

inline bool StreamTask::isFinished( )
{
    if ( alwaysFinished )
        return true;
    if(cudaEvent)
    {
        if ( cudaEvent->isFinished( ) )
        {
            alwaysFinished = true;
            return true;
        }
    }
    return false;
}

inline EventStream* StreamTask::getEventStream() const
{
    if(stream == nullptr)
        stream = Environment<>::get().TransactionManager().getEventStream(TASK_CUDA);
    return stream;
}

inline void StreamTask::setEventStream( EventStream* newStream )
{
    assert( newStream != NULL );
    assert( stream == NULL ); //it is only allowed to set a stream if no stream is set before
    this->stream = newStream;
}


inline void StreamTask::activate( )
{
    cudaEvent.reset(new CudaEvent(Environment<>::get().Manager().getEventPool().getNextEvent()));
    cudaEvent->recordEvent(this->stream->getCudaStream());
}

} //namespace PMacc
