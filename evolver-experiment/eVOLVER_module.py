import socket
import os.path
import shutil
import time
import pickle
import numpy as np
import numpy.matlib
import matplotlib
import scipy.signal
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from socketIO_client import SocketIO, BaseNamespace
from threading import Thread
import asyncio

plt.ioff()

dpu_evolver_ns = None
received_data = {}

# TODO: use proper async/await
wait_for_data = True
control = np.power(2,range(0,32))
current_chemo = [0] * 16
current_temps = [0] * 16
connected = False

class EvolverNamespace(BaseNamespace):
    def on_connect(self, *args):
        global connected
        print("Connected to eVOLVER as client")
        connected = True

    def on_disconnect(self, *args):
        global connected
        print("Discconected from eVOLVER as client")
        connected = False

    def on_reconnect(self, *args):
        global connected, stop_waiting
        print("Reconnected to eVOLVER as client")
        connected = True
        stop_waiting = True

    def on_dataresponse(self, data):
        global received_data, wait_for_data
        received_data = data
        wait_for_data = False

def read_data(vials, exp_name):
    global wait_for_data, received_data, current_temps, connected, stop_waiting
    od_cal = np.genfromtxt("OD_cal.txt", delimiter=',')
    temp_cal = np.genfromtxt("temp_calibration.txt", delimiter=',')
    save_path = os.path.dirname(os.path.realpath(__file__))

    wait_for_data = True
    stop_waiting = False
    dpu_evolver_ns.emit('data', {}, namespace='/dpu-evolver')
    while(wait_for_data):
        if not connected or stop_waiting:
            wait_for_data = False
            return
        pass

    od_data = received_data['OD']
    temp_data = received_data['temp']
    temps = []
    for x in vials:
        file_path =  "{0}/{1}/temp_config/vial{2}_tempconfig.txt".format(save_path,exp_name,x)
        temp_set_data = np.genfromtxt(file_path, delimiter=',')
        temp_set = temp_set_data[len(temp_set_data)-1][1]
        temp_set = int((temp_set - temp_cal[1][x])/temp_cal[0][x])
        temps.append(temp_set)
        try:
            od_data[x] = np.real(od_cal[2,x] - ((np.log10((od_cal[1,x]-od_cal[0,x])/(float(od_data[x]) - od_cal[0,x])-1))/od_cal[3,x]))
        except ValueError:
            print("OD Read Error")
            od_data[x] = 'NaN'
        try:
            temp_data[x] = (float(temp_data[x]) * temp_cal[0][x]) + temp_cal[1][x]
        except ValueError:
            print("Temp Read Error")
            temp_data[x]  = 'NaN'
    if not temps == current_temps:
        MESSAGE = list(temps)
        command = {'param':'temp', 'message':MESSAGE}
        dpu_evolver_ns.emit('command', command, namespace='/dpu-evolver')
        current_temps = temps
    return od_data, temp_data

def fluid_command(MESSAGE, vial, elapsed_time, pump_wait, exp_name, time_on, file_write):
    command = {'param':'pump', 'message':MESSAGE}
    dpu_evolver_ns.emit('command', command, namespace='/dpu-evolver')

    save_path = os.path.dirname(os.path.realpath(__file__))
    file_path =  "{0}/{1}/pump_log/vial{2}_pump_log.txt".format(save_path,exp_name,vial)
    if file_write == 'y':
        text_file = open(file_path,"a+")
        text_file.write("{0},{1}\n".format(elapsed_time, time_on))
        text_file.close()

def update_chemo(vials, exp_name, bolus_in_s, control):

    global current_chemo

    save_path = os.path.dirname(os.path.realpath(__file__))
    MESSAGE = {}
    for x in vials:
        file_path =  "{0}/{1}/chemo_config/vial{2}_chemoconfig.txt".format(save_path,exp_name,x)
        data = np.genfromtxt(file_path, delimiter=',')
        chemo_set = data[len(data)-1][2]
        if not chemo_set == current_chemo[x]:
            current_chemo[x] = chemo_set
            MESSAGE = {'pumps_binary':"{0:b}".format(control[x]), 'pump_time': bolus_in_s[x], 'efflux_pump_time': bolus_in_s[x] * 2, 'delay_interval': chemo_set, 'times_to_repeat': -1, 'run_efflux': 1}
            command = {'param': 'pump', 'message': MESSAGE}
            dpu_evolver_ns.emit('command', command, namespace = '/dpu-evolver')

def stir_rate (MESSAGE):
    command = {'param':'stir', 'message':MESSAGE}
    dpu_evolver_ns.emit('command', command, namespace='/dpu-evolver')

def parse_data(data, elapsed_time, vials, exp_name, file_name):
    save_path = os.path.dirname(os.path.realpath(__file__))
    if data == 'empty':
        print("%s Data Empty! Skipping data log...".format(file_name))
    else:
        for x in vials:
            file_path =  "{0}/{1}/{2}/vial{3}_{2}.txt".format(save_path,exp_name,file_name,x)
            text_file = open(file_path,"a+")
            text_file.write("{0},{1}\n".format(elapsed_time, data[x]))
            text_file.close()

def start_background_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

def run(evolver_ip, evolver_port):
    global dpu_evolver_ns
    socketIO = SocketIO(evolver_ip, evolver_port)
    dpu_evolver_ns = socketIO.define(EvolverNamespace, '/dpu-evolver')
    socketIO.wait()

def initialize_exp(exp_name, vials, evolver_ip, evolver_port):
    new_loop = asyncio.new_event_loop()
    t = Thread(target = start_background_loop, args = (new_loop,))
    t.daemon = True
    t.start()
    new_loop.call_soon_threadsafe(run, evolver_ip, evolver_port)

    if dpu_evolver_ns is None:
        print("Waiting for evolver connection...")

    while dpu_evolver_ns is None:
        pass

    save_path = os.path.dirname(os.path.realpath(__file__))
    dir_path =  "{0}/{1}".format(save_path,exp_name)
    exp_continue = input('Continue from existing experiment? (y/n): ')

    if exp_continue == 'n':

        start_time = time.time()
        if os.path.exists(dir_path):
            exp_overwrite = input('Directory aleady exists. Overwrite with new experiment? (y/n): ')
            if exp_overwrite == 'y':
                shutil.rmtree(dir_path)
            else:
                print('Change experiment name in custom_script.py and then restart...')
                exit() #exit

        os.makedirs("{0}/OD".format(dir_path))
        os.makedirs("{0}/temp".format(dir_path))
        os.makedirs("{0}/pump_log".format(dir_path))
        os.makedirs("{0}/temp_config".format(dir_path))
        os.makedirs("{0}/ODset".format(dir_path))
        os.makedirs("{0}/chemo_config".format(dir_path))

        for x in vials:
            OD_path =  "{0}/OD/vial{1}_OD.txt".format(dir_path,x)
            text_file = open(OD_path,"w").close()
            temp_path =  "{0}/temp/vial{1}_temp.txt".format(dir_path,x)
            text_file = open(temp_path,"w").close()
            tempconfig_path =  "{0}/temp_config/vial{1}_tempconfig.txt".format(dir_path,x)
            text_file = open(tempconfig_path,"w")
            text_file.write("Experiment: {0} vial {1}, {2}\n".format(exp_name, x, time.strftime("%c")))
            text_file.write("0,30\n")
            text_file.close()
            pump_path =  "{0}/pump_log/vial{1}_pump_log.txt".format(dir_path,x)
            text_file = open(pump_path,"w")
            text_file.write("Experiment: {0} vial {1}, {2}\n".format(exp_name, x, time.strftime("%c")))
            text_file.write("0,0\n")
            text_file.close()

            ODset_path =  "{0}/ODset/vial{1}_ODset.txt".format(dir_path,x)
            text_file = open(ODset_path,"w")
            text_file.write("Experiment: {0} vial {1}, {2}\n".format(exp_name, x, time.strftime("%c")))
            text_file.write("0,0\n")
            text_file.close()

            chemoconfig_path =  "{0}/chemo_config/vial{1}_chemoconfig.txt".format(dir_path,x)
            text_file = open(chemoconfig_path,"w")
            #text_file.write("Experiment: %s vial %d, %r\n" % (exp_name, x, time.strftime("%c")))
            text_file.write("0,0,0\n")
            text_file.write("0,0,0\n")
            text_file.close()

        OD_read = []
        temp_read = []
        while len(OD_read) == 0 and len(temp_read) == 0:
            OD_read, temp_read  = read_data(vials, exp_name)
            exp_blank = input('Calibrate vials to blank? (y/n): ')
            if exp_blank == 'y':
                OD_initial = OD_read
            else:
                OD_initial = np.zeros(len(vials))

    else:
        pickle_path =  "{0}/{1}/{2}.pickle".format(save_path,exp_name,exp_name)
        with open(pickle_path, 'rb') as f:
            loaded_var  = pickle.load(f)
        x = loaded_var
        start_time = x[0]
        OD_initial = x[1]

        # Restart chemostat pumps
        current_chemo = [0] * 16

    return start_time, OD_initial

def stop_all_pumps():
    command = {'param':'pump', 'message':'stop'}
    dpu_evolver_ns.emit('command', command, namespace='/dpu-evolver')
    print('All Pumps Stopped!')

def save_var(exp_name, start_time, OD_initial):
    save_path = os.path.dirname(os.path.realpath(__file__))
    pickle_path =  "{0}/{1}/{2}.pickle".format(save_path,exp_name,exp_name)
    with open(pickle_path, 'wb') as f:
        pickle.dump([start_time, OD_initial], f)

def restart_chemo():
    global current_chemo
    current_chemo = [0] * 16

if __name__ == '__main__':
    temp_data = update_temp('23423423')
    print(temp_data)
