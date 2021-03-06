#include <Windows.h>
#include <winhttp.h>


#include "WebSocket.h"

static HINTERNET hWebSocketHandle = NULL;
static HINTERNET hSessionHandle = NULL;
static HINTERNET hConnectionHandle = NULL;
static HINTERNET hRequestHandle = NULL;

unsigned long SendOnSocket(void* message, unsigned long size)
{
	return WinHttpWebSocketSend(
		hWebSocketHandle,
		WINHTTP_WEB_SOCKET_BINARY_MESSAGE_BUFFER_TYPE,
		message,
		size
	);
}

unsigned long ReceiveOnSocket(unsigned char* buffer, unsigned long buffer_size, unsigned long* size)
{
	WINHTTP_WEB_SOCKET_BUFFER_TYPE message_type;
	DWORD error;
	DWORD fragment_size;
	*size = 0;
	while (NO_ERROR == (error = WinHttpWebSocketReceive(hWebSocketHandle, buffer, buffer_size, &fragment_size, &message_type)) && message_type == WINHTTP_WEB_SOCKET_BINARY_FRAGMENT_BUFFER_TYPE)
	{
		buffer += fragment_size;
		buffer_size -= fragment_size;
		*size += fragment_size;
	}
	*size += fragment_size;
	return error + message_type;
}

/*
	credit to https://github.com/microsoft/Windows-classic-samples/blob/master/Samples/WinhttpWebsocket/cpp/WinhttpWebsocket.cpp
*/
unsigned long InitializeWebSocket(unsigned short port)
{
	//
	// Create session, connection and request handles.
	//

	hSessionHandle = WinHttpOpen(L"CnC WebSocket", WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, NULL, NULL, 0);
	if (hSessionHandle == NULL)
	{
		return GetLastError();
	}

	hConnectionHandle = WinHttpConnect(hSessionHandle, L"localhost", port, 0);
	if (hConnectionHandle == NULL)
	{
		return GetLastError();
	}

	hRequestHandle = WinHttpOpenRequest(hConnectionHandle, L"GET", L"", NULL, NULL, NULL, 0);
	if (hRequestHandle == NULL)
	{
		return GetLastError();
	}

	//
	// Request protocol upgrade from http to websocket.
	//
// #pragma prefast(suppress:6387, "WINHTTP_OPTION_UPGRADE_TO_WEB_SOCKET does not take any arguments.")
	if (FALSE == WinHttpSetOption(hRequestHandle, WINHTTP_OPTION_UPGRADE_TO_WEB_SOCKET, NULL, 0))
	{
		return GetLastError();
	}

	//
	// Perform websocket handshake by sending a request and receiving server's response.
	// Application may specify additional headers if needed.
	//

	if (FALSE == WinHttpSendRequest(hRequestHandle, WINHTTP_NO_ADDITIONAL_HEADERS, 0, NULL, 0, 0, 0))
	{
		return GetLastError();
	}

	if (FALSE == WinHttpReceiveResponse(hRequestHandle, 0))
	{
		return GetLastError();
	}

	//
	// Application should check what is the HTTP status code returned by the server and behave accordingly.
	// WinHttpWebSocketCompleteUpgrade will fail if the HTTP status code is different than 101.
	//

	hWebSocketHandle = WinHttpWebSocketCompleteUpgrade(hRequestHandle, 0UL);
	if (hWebSocketHandle == NULL)
	{
		return GetLastError();
	}

	//
	// The request handle is not needed anymore. From now on we will use the websocket handle.
	//

	WinHttpCloseHandle(hRequestHandle);
	hRequestHandle = NULL;

	return 0;
}


void DestroyWebSocket()
{
	//
	// Gracefully close the connection.
	//

	WinHttpWebSocketClose(hWebSocketHandle, WINHTTP_WEB_SOCKET_SUCCESS_CLOSE_STATUS, NULL, 0);

	if (hRequestHandle != NULL)
	{
		WinHttpCloseHandle(hRequestHandle);
		hRequestHandle = NULL;
	}

	if (hWebSocketHandle != NULL)
	{
		WinHttpCloseHandle(hWebSocketHandle);
		hWebSocketHandle = NULL;
	}

	if (hConnectionHandle != NULL)
	{
		WinHttpCloseHandle(hConnectionHandle);
		hConnectionHandle = NULL;
	}

	if (hSessionHandle != NULL)
	{
		WinHttpCloseHandle(hSessionHandle);
		hSessionHandle = NULL;
	}
}
