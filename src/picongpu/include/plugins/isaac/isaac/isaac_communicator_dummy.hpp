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
		void serverSendFrame(void* ptr,const isaac_uint width,const isaac_uint height,const isaac_uint depth)
		{
			//First the size
			uint32_t count = width*height*depth;
			#if ISAAC_JPEG == 1
				struct jpeg_compress_struct cinfo;
				struct jpeg_error_mgr jerr;
				jpeg_destination_mgr dest;
				dest.init_destination = &isaac_init_destination;
				dest.empty_output_buffer = &isaac_jpeg_empty_output_buffer;
				dest.term_destination = &isaac_jpeg_term_destination;
				cinfo.err = jpeg_std_error(&jerr);
				jpeg_create_compress(&cinfo);
				cinfo.dest = &dest;
				std::vector<char> jpeg_buffer;
				jpeg_buffer.resize( count );
				cinfo.dest->next_output_byte = (JOCTET*)(jpeg_buffer.data());
				cinfo.dest->free_in_buffer = count;
				cinfo.image_width = width;
				cinfo.image_height = height;
				cinfo.input_components = depth;
				cinfo.in_color_space = JCS_CMYK;
				jpeg_set_defaults(&cinfo);
				jpeg_set_quality(&cinfo, jpeg_quality, false);
				jpeg_start_compress(&cinfo, TRUE);
				while (cinfo.next_scanline < cinfo.image_height)
				{
					JSAMPROW row_pointer[1];
					row_pointer[0] = & ((JSAMPROW)ptr)[cinfo.next_scanline * width * depth];
					(void) jpeg_write_scanlines(&cinfo, row_pointer, 1);
				}
				jpeg_finish_compress(&cinfo);
				count -= cinfo.dest->free_in_buffer;
				ptr = jpeg_buffer.data();
				jpeg_destroy_compress(&cinfo);
			#endif

			using namespace boost::archive::iterators;
			std::stringstream payload;
			typedef
				base64_from_binary
				<
					transform_width
					<
						const unsigned char *,
						6,
						8
					>
				>
				base64_text; // compose all the above operations in to a new iterator

			std::copy(
				base64_text( (char*)ptr ),
				base64_text( (char*)ptr + count),
				boost::archive::iterators::ostream_iterator<char>(payload)
			);

			#if ISAAC_JPEG == 1
				char header[] = "{\"payload\": \"data:image/jpeg;base64,";
			#else
				char header[] = "{\"payload\": \"data:image/raw-rgba;base64,";
			#endif
			char footer[] = "\"}";
			int hl = strlen(header);
			int pl = payload.str().length();
			int fl = strlen(footer);
			char* message = (char*)malloc(hl+pl+fl+1);
			memcpy(  message        ,header,hl);
			memcpy(&(message[hl   ]),payload.str().c_str(),pl);
			memcpy(&(message[hl+pl]),footer,fl+1); //with 0
			serverSend( message, false, true );
			free(message);
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
