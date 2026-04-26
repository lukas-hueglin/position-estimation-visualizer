import wx

import pyrr
import numpy as np

from OpenGL.GL import *
from OpenGL.GLU import *

from rendering.figures import Cube
from rendering.renderer import Renderer

class BroadRenderer(Renderer):
    def __init__(self, data_handler):
        super().__init__(data_handler)

        # OpenGL variable
        self.view_matrix = pyrr.matrix44.create_look_at(
            np.array([3.0, 3.0, 3.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0])).T
        self.proj_matrix = pyrr.matrix44.create_perspective_projection(
            45, (800/600), 0.1, 200.0).T

        # add a slider
        self.slider = wx.Slider(self.canvas, 1, value=0, minValue=0, maxValue=self.data_handler.n_samples-2, size=(800, 25))
        self.slider.SetBackgroundColour(wx.Colour(46, 45, 64))
        self.Bind(wx.EVT_SLIDER, self.on_slider_change, id=1)

    def init_opengl(self):
        super().init_opengl()
    
        # add vertex arrays
        self.validationVertexArray = glGenVertexArrays(1)
        self.init_buffer(self.validationVertexArray, Cube, 100)

        self.predictionVertexArray = glGenVertexArrays(1)
        self.init_buffer(self.predictionVertexArray, Cube, 255)

    def draw(self):
        glUseProgram(self.gl_program)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # call parent draw function
        super().draw()

        # draw cubes
        q_pred = self.data_handler.get_result('x', self.slider.GetValue())
        self.draw_buffer(self.predictionVertexArray, Cube, (0, 0, 0), q_pred, (1, 1, 1))

        q_val = self.data_handler.get_validation(self.slider.GetValue())
        self.draw_buffer(self.validationVertexArray, Cube, (0, 0, 0), q_val, (1.5, 1.5, 1.5))
        
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

    def on_size(self, event):
        super().on_size(event)
    
        size = self.canvas.GetClientSize()
        self.slider.SetSize(size.width, 25)

    def on_slider_change(self, event):
        self.on_paint(event)