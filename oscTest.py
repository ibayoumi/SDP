# OSC Test for Streaming GSR Data
from osc4py3.as_eventloop import*
from osc4py3 import oscmethod as osm


def handlerfunction(s, x, y):
    # Will receive message data unpacked in s, x, y
    print(s)
    print(x)
    print(y)
    pass


def handlerfunction2(address, s, x, y):
    # Will receive message address, and message data flattened in s, x, y
    pass


# Start the system\
osc_startup()

# 47.148.255.12 --> home IP addr
# 192.168.255.255 --> IP addr defined in eSense
# Note: Change the IP address accordingly
# Make server channels to receive packets
osc_udp_server("192.168.254.125", 8000, "aservername")

# Associate Python functions with message address patterns, using defaults argument
# scheme OSCARG_DATAUNPACK
osc_method("/test/*", handlerfunction)

# Too, but request the message address pattern before in argscheme
osc_method("/test/*", handlerfunction2, argscheme=osm.OSCARG_ADDRESS + osm.OSCARG_DATAUNPACK)

# Periodically call osc4py3 processing method in your event loop
finished = False
while not finished:
    osc_process()
    print('here')
    finished = True

# Properly close the system
osc_terminate()
