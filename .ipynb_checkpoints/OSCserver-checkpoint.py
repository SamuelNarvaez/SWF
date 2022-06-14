import queue
import time
import random
import fileinput
import sys

from trimesh import *
from swf import *
from optimal import *
from utils import *
from constants import *

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
    print('initializing model . . .')
    
    method = sys.argv[1]
    
    if method == '704base':
        print('704base')
        #for the subdivision mesh based on 7.0.4
        model = OptimalSWF(vertices704,faces704).model 
        encoder = model.phi2s[0]
    
    elif method == 'transcoding':
        print(f'transcoding {sys.argv[2]}')
        #for the transcoding mesh
        key = transcoding_precomputed_coeffs
        base = Trimesh(v_3_0,f_3_0,ALPHA=key[0][0],BETA=key[0][1],GAMMA=key[0][2])
        first = base.manual_subdivide(v_5_0,f_5_0,ALPHA=key[1][0],BETA=key[1][1],GAMMA=key[1][2])
        second = first.manual_subdivide(v_5_2,f_5_2,ALPHA=key[2][0],BETA=key[2][1],GAMMA=key[2][2])
        third = second.manual_subdivide(v_7_4,f_7_4,ALPHA=key[3][0],BETA=key[3][1],GAMMA=key[3][2])
        fourth = third.manual_subdivide(v_9_6,f_9_6,ALPHA=key[4][0],BETA=key[4][1],GAMMA=key[4][2])
        fifth = fourth.manual_subdivide(v_11_8,f_11_8,ALPHA=key[5][0],BETA=key[5][1],GAMMA=key[5][2])
        opt_meshset = [first,second,third,fourth,fifth]
        model = SWF(base,2,meshset=opt_meshset)
        encoder = model.phi2s[int(sys.argv[2])]
    
    print('built model!')
    
    np.savetxt('encoder.txt',encoder,newline=';\n')
    for line in fileinput.input('encoder.txt',inplace=True):
        sys.stdout.write('%d, %s'%(fileinput.filelineno(), line))
    
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