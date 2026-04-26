import time
import threading

import pyrr
import numpy as np

from OpenGL.GL import *
from OpenGL.GLU import *


from rendering.figures import Cube
from rendering.renderer import Renderer

class LiveRenderer(Renderer):
    def __init__(self, data_handler):
        super().__init__(data_handler)

        # OpenGL variable
        self.view_matrix = pyrr.matrix44.create_look_at(
            np.array([0.0, -3.0, 3.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0])).T
        self.proj_matrix = pyrr.matrix44.create_perspective_projection(
            45, (800/600), 0.1, 200.0).T

        # add self updating functionality
        def loop():
            while True:
                self.Refresh()
                time.sleep(1.0/30)

        self.thread = threading.Thread(target=loop)
        self.thread.start()

    def __del__(self):
        self.thread.join()


    def init_opengl(self):
        super().init_opengl()
    
        # add vertex arrays
        self.rotationVertexArray = glGenVertexArrays(1)
        self.init_buffer(self.rotationVertexArray, Cube, 255)

    
    def draw(self):
        glUseProgram(self.gl_program)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # call parent draw function
        super().draw()

        quat = self.data_handler.get_result('x')

        theta = np.deg2rad(45)
        init_X = np.array([np.cos(theta/2), 0*np.sin(theta/2), 0*np.sin(theta/2), 1*np.sin(theta/2)])
        self.draw_buffer(self.rotationVertexArray, Cube, (0, 0, 0), quat, (1, 1, 1))
        
        # draw lines
        #z = self.data_handler.get_result('z', self.slider.GetValue())
        #hx = self.data_handler.get_result('h(x)', self.slider.GetValue())
        #m = self.data_handler.mag_dc[self.slider.GetValue()]

        #self.draw_line((0, 0, 0), quat_transform_point(q_pred, m))
        #self.draw_line((0, 0, 0),  m)
        
        #self.draw_line((0, 0, 0), 100*quat_transform_point(quat_conj(q_pred), z[:3]))
        #self.draw_line((0, 0, 0), 100*quat_transform_point(quat_conj(q_pred), hx[:3]))
        #self.draw_line((0, 0, 0), 100*quat_transform_point(quat_conj(q_pred), z[3:]))
        #self.draw_line((0, 0, 0), 100*quat_transform_point(quat_conj(q_pred), hx[3:]))

        glUseProgram(0)

        self.canvas.SwapBuffers()

        