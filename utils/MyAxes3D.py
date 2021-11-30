# -*- coding: utf-8 -*-

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

'''
import mpl_toolkits.mplot3d.axes3d as p3

from MyAxes3D import MyAxes3D

fig = plt.figure()
ax = p3.Axes3D(fig)
ax = fig.add_axes(MyAxes3D(ax, 'l'))
ax.set_proj_type('ortho')
ax.scatter(X[0,:], X[1,:], y_true, marker='o', c='r', s=50)
ax.set_xlabel('X_0', fontsize=15)
ax.set_ylabel('X_1', fontsize=15)
ax.set_zlabel('Y', fontsize=15)
'''

class MyAxes3D(axes3d.Axes3D):

    def __init__(self, baseObject, sides_to_draw):
        self.__class__ = type(baseObject.__class__.__name__,
                              (self.__class__, baseObject.__class__),
                              {})
        self.__dict__ = baseObject.__dict__
        self.sides_to_draw = list(sides_to_draw)
        self.mouse_init()

    def set_some_features_visibility(self, visible):
        for t in self.w_zaxis.get_ticklines() + self.w_zaxis.get_ticklabels():
            t.set_visible(visible)
        self.w_zaxis.line.set_visible(visible)
        self.w_zaxis.pane.set_visible(visible)
        self.w_zaxis.label.set_visible(visible)

    def draw(self, renderer):
        # set visibility of some features False 
        self.set_some_features_visibility(False)
        # draw the axes
        super(MyAxes3D, self).draw(renderer)
        # set visibility of some features True. 
        # This could be adapted to set your features to desired visibility, 
        # e.g. storing the previous values and restoring the values
        self.set_some_features_visibility(True)

        zaxis = self.zaxis
        draw_grid_old = zaxis.axes._draw_grid
        # disable draw grid
        zaxis.axes._draw_grid = False

        tmp_planes = zaxis._PLANES

        if 'l' in self.sides_to_draw :
            # draw zaxis on the left side
            zaxis._PLANES = (tmp_planes[2], tmp_planes[3],
                             tmp_planes[0], tmp_planes[1],
                             tmp_planes[4], tmp_planes[5])
            zaxis.draw(renderer)
        if 'r' in self.sides_to_draw :
            # draw zaxis on the right side
            zaxis._PLANES = (tmp_planes[3], tmp_planes[2], 
                             tmp_planes[1], tmp_planes[0], 
                             tmp_planes[4], tmp_planes[5])
            zaxis.draw(renderer)

        zaxis._PLANES = tmp_planes

        # disable draw grid
        zaxis.axes._draw_grid = draw_grid_old

if __name__ == '__main__':
    import mpl_toolkits.mplot3d.axes3d as p3

    # from MyAxes3D import MyAxes3D

    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax = fig.add_axes(MyAxes3D(ax, 'l'))
    ax.set_proj_type('ortho')
    ax.scatter(1, -1, 1, marker='o', c='r', s=50)