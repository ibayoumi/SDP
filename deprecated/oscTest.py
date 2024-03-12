# OSC Test for Streaming GSR Data
from osc4py3.as_eventloop import*
from osc4py3 import oscmethod as osm
import logging
import threading


logging.basicConfig(format='%(asctime)s - %(threadName)s ø %(name)s - ''%(levelname)s - %(message)s')
logger = logging.getLogger("osc")
logger.setLevel(logging.DEBUG)

def handlerfunction(*args):
    for arg in args:
        print(arg)


def handlerfunction2(address, s, x, y):
    # Will receive message address, and message data flattened in s, x, y
    print("handler2 called")


# The function called to handle matching messages.
hlock = threading.Lock()
hcount = 0    

def handlerfunction3(*args):
    global hlock, hcount
    with hlock:
        hcount += 1
        if logger and logger.isEnabledFor(logging.INFO):
            logger.info("##### %d handler function called with: %r",
                    hcount, args)
        else:
            print("##### {} handler function called with: {!r}".format(hcount
            , args))
    # print('handler 3')
    # global hlock, hcount
    # with hlock:
    #     print("##### {} handler function called with: {!r}".format(hcount, args))


# Start the system\
# osc_startup(logger=logger)
osc_startup()


# Note: Change the IP address accordingly
# Make server channels to receive packets

# To use, make sure hotspot is on. Connect laptop to hotspot. Open cmd on laptop, type
# 'ipconfig /all', and use IPv4 under WiFi for both client and server. Define any port.
IP = '192.168.159.207'
PORT = 8000
osc_udp_client(IP, PORT, "udplisten")
osc_udp_server("192.168.159.207", PORT, "udpclient")

# Associate Python functions with message address patterns, using defaults argument
"""
 Addr pattern is '/<variable>'. There are two you can retrieve
 (1) '/time' gives you the timestamp
 (2) '/<variable>' where 'variable' is the name you set on OSC settings in eSense app
      For example, default variable is set to '/edaMikroS'.
"""
# osc_method("/edaMikroS", handlerfunction3, argscheme=osm.OSCARG_DATAUNPACK)
osc_method("/edaMikroS", handlerfunction, argscheme=osm.OSCARG_DATAUNPACK)

# Periodically call osc4py3 processing method in your event loop
finished = False
first = True
while not finished:
    try:
        osc_process()
    except KeyboardInterrupt:
        finished = True

# Properly close the system
osc_terminate()

