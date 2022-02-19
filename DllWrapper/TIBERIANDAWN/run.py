import sys
import struct

import tornado.web
import tornado.websocket
import tornado.ioloop


class MainHandler(tornado.websocket.WebSocketHandler):
    def on_message(self, message):
        print(message)

    def open(self):
        self.set_nodelay(True)
        buffer = b''

        if len(sys.argv) > 3:
            buffer += struct.pack('L256s', 1, sys.argv[3].encode('utf8'))

        buffer += struct.pack(
            'L256s256s', 2, sys.argv[1].encode('utf8'), sys.argv[2].encode('ascii')
        )
        self.write_message(buffer, binary=True)

    def on_close(self):
        print('Bye!')


def main():
    port = 8889
    application = tornado.web.Application([(r"/", MainHandler)])
    application.listen(port)
    tornado.ioloop.IOLoop.current().start()


if __name__ == '__main__':
    main()
