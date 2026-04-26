from OpenGL.GL import *
from OpenGL.GLU import *

import numpy as np

class Cube:
    face_type = GL_QUADS

    def get_vertices():
        return np.array([
            0.5, 0.5, 0.5, # Front Face
            0.5,-0.5, 0.5,
            -0.5,-0.5, 0.5,
            -0.5, 0.5, 0.5,
            
            0.5, 0.5, 0.5, # Right Face
            0.5, 0.5,-0.5,
            0.5,-0.5,-0.5,
            0.5,-0.5, 0.5,

            0.5, 0.5, 0.5, # Top Face
            -0.5, 0.5, 0.5,
            -0.5, 0.5,-0.5,
            0.5, 0.5,-0.5,

            0.5, 0.5,-0.5, # Rear Face
            0.5,-0.5,-0.5,
            -0.5,-0.5,-0.5,
            -0.5, 0.5,-0.5,
            
            -0.5, 0.5, 0.5, # Left Face
            -0.5, 0.5,-0.5,
            -0.5,-0.5,-0.5,
            -0.5,-0.5, 0.5,

            0.5,-0.5, 0.5, # Bottom Face
            -0.5,-0.5, 0.5,
            -0.5,-0.5,-0.5,
            0.5,-0.5,-0.5,

        ], dtype='float32')

    def get_color(alpha):
        return np.array([
            19.0,   94.0,   242.0, alpha,
            19.0,   94.0,   242.0, alpha,
            19.0,   94.0,   242.0, alpha,
            19.0,   94.0,   242.0, alpha,
            242.0,  66.0,   128.0, alpha,
            242.0,  66.0,   128.0, alpha,
            242.0,  66.0,   128.0, alpha,
            242.0,  66.0,   128.0, alpha,
            41.0,   217.0,  152.0, alpha,
            41.0,   217.0,  152.0, alpha,
            41.0,   217.0,  152.0, alpha,
            41.0,   217.0,  152.0, alpha,
            19.0,   94.0,   242.0, alpha,
            19.0,   94.0,   242.0, alpha,
            19.0,   94.0,   242.0, alpha,
            19.0,   94.0,   242.0, alpha,
            242.0,  66.0,   128.0, alpha,
            242.0,  66.0,   128.0, alpha,
            242.0,  66.0,   128.0, alpha,
            242.0,  66.0,   128.0, alpha,
            41.0,   217.0,  152.0, alpha,
            41.0,   217.0,  152.0, alpha,
            41.0,   217.0,  152.0, alpha,
            41.0,   217.0,  152.0, alpha,
        ], dtype='float32') / 255

class Sphere:
    face_type = GL_TRIANGLES

    def get_vertices():
        num_segments = 20
        vertices = []

        for i in range(num_segments):
            theta1 = i * 2 * np.pi / num_segments
            theta2 = (i + 1) * 2 * np.pi / num_segments

            for j in range(num_segments):
                phi1 = j * np.pi / num_segments
                phi2 = (j + 1) * np.pi / num_segments

                # Vertices
                v1 = [np.sin(phi1) * np.cos(theta1),
                    np.sin(phi1) * np.sin(theta1),
                    np.cos(phi1)]
                v2 = [np.sin(phi1) * np.cos(theta2),
                    np.sin(phi1) * np.sin(theta2),
                    np.cos(phi1)]
                v3 = [np.sin(phi2) * np.cos(theta2),
                    np.sin(phi2) * np.sin(theta2),
                    np.cos(phi2)]
                v4 = [np.sin(phi2) * np.cos(theta1),
                    np.sin(phi2) * np.sin(theta1),
                    np.cos(phi2)]

                # Adding vertices to the list
                vertices.extend([v1, v2, v3, v1, v3, v4])

        return np.ndarray.flatten(np.array(vertices, dtype='float32'))
    
    def get_color(alpha):
        num_segments = 20
        return np.ndarray.flatten(np.array([[100, 100, 100, alpha] for _ in range((num_segments**2)*6)], dtype='float32'))/255
