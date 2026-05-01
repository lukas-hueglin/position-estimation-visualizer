import wx
import wx.glcanvas

import pyrr

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL import shaders

from rendering.figures import Cube
from helpers.quat_math import *
from helpers.data_handler import*


class Renderer(wx.Frame):
    def __init__(self, data_handler):
        super().__init__(parent=None, title='Position Estimation - Visualisation Tool', size=(800, 600))
        self.data_handler = data_handler

        self.initialized = False
 
        attributes = (wx.glcanvas.WX_GL_RGBA,
                      wx.glcanvas.WX_GL_DOUBLEBUFFER,
                      wx.glcanvas.WX_GL_DEPTH_SIZE, 24)
 
        self.canvas = wx.glcanvas.GLCanvas(self, attribList=attributes)
        self.context = wx.glcanvas.GLContext(self.canvas)
 
        self.canvas.Bind(wx.EVT_SIZE, self.on_size)
        self.canvas.Bind(wx.EVT_PAINT, self.on_paint)

    
    def init_opengl(self):
        self.initialized = True

        # open shader source
        with open('src\\rendering\\VertexShader.vert', 'r') as vert_file:
            vert_source = vert_file.read()
        with open('src\\rendering\\FragmentShader.frag', 'r') as frag_file:
            frag_source = frag_file.read()

        self.gl_program = shaders.compileProgram(
            shaders.compileShader(vert_source, GL_VERTEX_SHADER),
            shaders.compileShader(frag_source, GL_FRAGMENT_SHADER)
        )

        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)

        glClearColor((46/255),(45/255),(64/255),1)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_BLEND)
        
        # init vertex arrays
        self.orientationVertexArray = glGenVertexArrays(1)
        self.init_buffer(self.orientationVertexArray, Cube, 150)

        
    def init_shaders(self):
        vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)

        with open('VertexShader.vert', 'r') as vert_file:
            vert_source = vert_file.read()
        with open('FragmentShader.frag', 'r') as frag_file:
            frag_source = frag_file.read()

        glShaderSource(vertex_shader, vert_source)
        glShaderSource(fragment_shader, frag_source)

        glCompileShader(vertex_shader)
        if not glGetShaderiv(vertex_shader, GL_COMPILE_STATUS):
            info_log = glGetShaderInfoLog(vertex_shader)
            print ('Compilation Failure for ' + vert_source + ' shader:\n' + info_log)

        glCompileShader(fragment_shader)
        if not glGetShaderiv(fragment_shader, GL_COMPILE_STATUS):
            info_log = glGetShaderInfoLog(fragment_shader)
            print ('Compilation Failure for ' + frag_source + ' shader:\n' + info_log)

        glAttachShader(self.gl_program, vertex_shader)
        glAttachShader(self.gl_program, fragment_shader)

        glLinkProgram(self.gl_program)

        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)


    def init_buffer(self, vertexArray, obj, alpha):
        glBindVertexArray(vertexArray)

        self.posBuffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.posBuffer)
        glBufferData(GL_ARRAY_BUFFER, obj.get_vertices(), GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        self.colBuffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.colBuffer)
        glBufferData(GL_ARRAY_BUFFER, obj.get_color(alpha), GL_STATIC_DRAW)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)

        glBindVertexArray(0)


    def draw_buffer(self, vertexArray, obj, trans, rot, scale, scr_offset=(0, 0)):
        glBindVertexArray(vertexArray)

        # calculate mvp matrix
        model_matrix = self.transform(trans, rot, scale).T
        mvp = (self.proj_matrix @ self.view_matrix @ model_matrix).T

        # send mvp matrix to gpu
        trans_uniform = glGetUniformLocation(self.gl_program, 'mvp')
        glUniformMatrix4fv(trans_uniform, 1, GL_FALSE, mvp)

        # send screen_space_trans matrix to gpus
        scr_offset_uniform = glGetUniformLocation(self.gl_program, 'scr_offset')
        glUniform2f(scr_offset_uniform, *scr_offset)

        # draw vertex array
        glDrawArrays(obj.face_type, 0, len(obj.get_vertices()))

        glBindVertexArray(0)


    def draw_line(self, start, end, reset=True):
        if reset:
            # calculate mvp matrix
            model_matrix = self.transform((0, 0, 0), (1, 0, 0, 0), (1, 1, 1)).T
            mvp = (self.proj_matrix @ self.view_matrix @ model_matrix).T

            # send mvp matrix to gpu
            trans_uniform = glGetUniformLocation(self.gl_program, 'mvp')
            glUniformMatrix4fv(trans_uniform, 1, GL_FALSE, mvp)

            # send screen_space_trans matrix to gpus
            scr_offset_uniform = glGetUniformLocation(self.gl_program, 'scr_offset')
            glUniform2f(scr_offset_uniform, 0, 0)

        glBegin(GL_LINES)
        glColor4f(255, 0, 0, 255)
        glVertex3f(*start)
        glVertex3f(*end)
        glEnd()
        

    def on_paint(self, event):
        if not self.initialized:
            self.init_opengl()
 
        self.draw()


    def draw(self):
        # draw orientation cube
        self.draw_buffer(self.orientationVertexArray, Cube, (0, 0, 0), (1, 0, 0, 0), (0.3, 0.3, 0.3), scr_offset=(0.82, 0.75))

        # draw coordinate system
        #self.draw_line((0, 0, 0),  (10, 0, 0))
        #self.draw_line((0, 0, 0),  (0, 10, 0))
        #self.draw_line((0, 0, 0),  (0, 0, 10))


    def on_size(self, event):
        self.canvas.SetCurrent(self.context)
 
        size = self.canvas.GetClientSize()
        glViewport(0, 0, size.width, size.height)

        self.proj_matrix = pyrr.matrix44.create_perspective_projection(
            45, (size.width/size.height), 0.1, 200.0).T
 
        self.canvas.Refresh(False)


    def transform(self, trans, rot, scale):
        trans_matrix = pyrr.matrix44.create_from_translation(trans)
        rot_matrix = pyrr.matrix44.create_from_quaternion(pyrr.quaternion.create(-rot[1], -rot[2], -rot[3],rot[0]))
        scale_matrix = pyrr.matrix44.create_from_scale(scale)

        return rot_matrix @ scale_matrix @ trans_matrix