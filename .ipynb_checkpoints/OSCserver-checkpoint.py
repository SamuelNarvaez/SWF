import queue
import time
import random

from trimesh import *
from swf import *
from optimal import *
from utils import *
import numpy as np
import soundfile as sf
from scipy import spatial

from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.dispatcher import Dispatcher
from pythonosc.udp_client import SimpleUDPClient


if __name__ == '__main__':

    # Lists for storing received values
    pos = [0]
    quitFlag = [False]

    # ------------------ OSC ips / ports ------------------ #
    # connection parameters
    ip = "127.0.0.1"
    receiving_from_pd_port = 1415
    sending_to_pd_port = 1123

    # ----------------------------------------------------------

    # ------------------ OSC Receiver from Pd ------------------ #
    # create an instance of the osc_sender class above
    py_to_pd_OscSender = SimpleUDPClient(ip, sending_to_pd_port)
    # ---------------------------------------------------------- #

    # ------------------ OSC Receiver from Pd ------------------ #
    # dispatcher is used to assign a callback to a received osc message
    # in other words the dispatcher routes the osc message to the right action using the address provided
    dispatcher = Dispatcher()

    # define the handler for messages starting with /position]
    def pos_message_handler(address, *args):
        pos[0] = np.array(args)

    # pass the handlers to the dispatcher
    dispatcher.map("/position*", pos_message_handler)

    # you can have a default_handler for messages that don't have dedicated handlers
    def default_handler(address, *args):
        print(f"No action taken for message {address}: {args}")
    dispatcher.set_default_handler(default_handler)

    # python-osc method for establishing the UDP communication with pd
    server = BlockingOSCUDPServer((ip, receiving_from_pd_port), dispatcher)
    # ---------------------------------------------------------- #

    # ------------------ Interpolation GENERATION  ------------------ #
    layout704=np.array([[1,np.pi/6,np.pi/2],[1,-np.pi/6,np.pi/2],[1,0,np.pi/2],[1,np.pi/2,np.pi/2],[1,-np.pi/2,np.pi/2],[1,3*np.pi/4,np.pi/2],[1,-3*np.pi/4,np.pi/2],[1,np.pi/4,np.pi/4],[1,-np.pi/4,np.pi/4],[1,3*np.pi/4,np.pi/4],[1,-3*np.pi/4,np.pi/4]])
    vertices704 = np.apply_along_axis(lambda x: toCartesian(x),1,layout704)
    faces704 = np.array([[6,4,10],[10,4,8],[8,4,1],[8,1,2],[8,7,2],[7,2,0],[7,0,3],[7,3,9],[9,3,5],[10,7,9],[10,8,7],[10,6,5],[10,9,5]])
    print('initializing model . . .')
    model = OptimalSWF(vertices704,faces704).model
    print('built model!')
    triangles = model.meshes[-1].vertices[model.meshes[-1].faces]

    while (quitFlag[0] is False):
        server.handle_request()
        print(f"Received position value: {pos[0]}")
        #rad = np.radians(pos[0])
        #loc = toCartesian(np.hstack((1,rad)))
        loc = pos[0].reshape((1,3))
        fine = model.interpolate(loc)

        # 3. Send Notes to pd (send pitch last to ensure syncing)
        py_to_pd_OscSender.send_message("/interpolation", (fine.reshape(-1).tolist()))
        print(f'interpolation: {fine[fine!=0]}')

    # ---------------------------------------------------------- #