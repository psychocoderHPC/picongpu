/* This file is part of ISAAC.
 *
 * ISAAC is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * ISAAC is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with ISAAC.  If not, see <www.gnu.org/licenses/>. */

#pragma once

#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <netdb.h>
#include <unistd.h>

#if ISAAC_JPEG == 1
    #include <jpeglib.h>
#endif

#include "isaac_macros.hpp"

#include <boost/archive/iterators/base64_from_binary.hpp>
#include <boost/archive/iterators/transform_width.hpp>
#include <boost/archive/iterators/ostream_iterator.hpp>
#include <sstream>

namespace isaac
{

class IsaacCommunicator
{
	public:
		IsaacCommunicator(const std::string url,const isaac_uint port) :
			id(0),
			server_id(0),
			url(url),
			port(port),
			jpeg_quality(90)
		{
			pthread_mutex_init (&deleteMessageMutex, NULL);
		}
		json_t* getLastMessage()
		{
			json_t* result = NULL;
			pthread_mutex_lock(&deleteMessageMutex);
			if (!messageList.empty())
			{
				result = messageList.front();
				messageList.pop_front();
			}
			pthread_mutex_unlock(&deleteMessageMutex);
			return result;
		}
		isaac_int serverConnect()
		{
			return 0;
		}

		isaac_int serverSend(char * content, bool starting = true, bool finishing = false)
		{
			return 0;
		}

		#if ISAAC_JPEG == 1
			static void isaac_init_destination(j_compress_ptr cinfo)
			{
			}
			static boolean isaac_jpeg_empty_output_buffer(j_compress_ptr cinfo)
			{
				return true;
			}
			static void isaac_jpeg_term_destination(j_compress_ptr cinfo)
			{
			}
		#endif
		void setJpegQuality(isaac_uint jpeg_quality)
		{
			if (jpeg_quality > 100)
				jpeg_quality = 100;
			this->jpeg_quality = jpeg_quality;
		}
		void serverDisconnect()
		{
		}
		~IsaacCommunicator()
		{
			usleep(100000); //100ms
			pthread_cancel(readThread);
			pthread_mutex_destroy(&deleteMessageMutex);
		}
		void addMessage( json_t* content )
		{
			pthread_mutex_lock(&deleteMessageMutex);
			messageList.push_back( content );
			pthread_mutex_unlock(&deleteMessageMutex);
		}
	private:
		isaac_uint id;
		isaac_uint server_id;
		std::string url;
		isaac_uint port;
		isaac_uint jpeg_quality;
		std::list<json_t*> messageList;
		pthread_mutex_t deleteMessageMutex;
		pthread_t readThread;
};

} //namespace isaac;
