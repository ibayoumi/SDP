import PySimpleGUI as sg

#import subprocess
import os
from multiprocessing import Process

def script1():
    os.system("D:/CARLA_0.9.14/WindowsNoEditor/CarlaUE4.exe")
def script2():
    os.system("python App_Zephyr_main/main.py")
def script3():
    os.system("python manual_wheel_zephyr.py")
    


if __name__ == '__main__':
    p1 = Process(target=script1)
    p2 = Process(target=script2)
    p3 = Process(target=script3)

    #Define the window's contents
    layout = [[sg.Text("What's your city name?")],
            [sg.Input(key='-INPUT-')],
            [sg.Text(size=(40,1), key='-OUTPUT-')],
            [sg.Button('Setup Carla Environment'), sg.Button('Setup Belt Connection'), sg.Button('Run Program'), sg.Button('Quit')]]
    
    #Create the window
    window = sg.Window('Window Title', layout)
    
    #Display and interact with the Window using an Event Loop
    while True:
        event, values = window.read()
        # See if user wants to quit or window was closed
        if event == sg.WINDOW_CLOSED or event == 'Quit':
            break
        elif event == 'Setup Carla Environment':
            #p = subprocess.Popen("python App_Zephyr_main/main.py", start_new_session=True)
            #shell=True, stdout=subprocess.PIPE
            #output, err = p.communicate()
            #print("                             " + output)
            p1.start()
        #subprocess.Popen("D:/CARLA_0.9.14/WindowsNoEditor/CarlaUe4.exe", start_new_session=True)
        elif event == 'Setup Belt Connection':
            p2.start()
        elif event == 'Run Program':
            #subprocess.Popen("python manual_wheel_zephyr.py", start_new_session=True)
            p3.start()
            #p1.join()
            #p1.join()
            #p3.join()
        # Output a message to the window
        window['-OUTPUT-'].update('Welcome to ' + values['-INPUT-'] + "! Thanks for trying PySimpleGUI")
    
    # Finish up by removing from the screen
    p3.close()
    p2.close()
    p1.close()
    window.close()
