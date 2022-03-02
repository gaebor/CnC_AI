#ifdef __cplusplus
extern "C" {
#endif

unsigned long InitializeWebSocket(unsigned short port);
void DestroyWebSocket();

unsigned long SendOnSocket(void* message, unsigned long size);
unsigned long ReceiveOnSocket(unsigned char* buffer, unsigned long buffer_size, unsigned long* size);

#ifdef __cplusplus
}
#endif