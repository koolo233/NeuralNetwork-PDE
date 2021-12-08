# !/usr/bin python3                                 
# encoding    : utf-8 -*-                            
# @author     : Zijiang Yang                                   
# @file       : PlotUtils.py
# @Time       : 2021/12/7 16:28
import os
import shutil

import imageio
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def heatmap_plot_func(data_matrix,
                      x_value_list,
                      y_value_list,
                      draw_size,
                      title="",
                      xlabel="x",
                      ylabel="y",
                      figsize=(5, 5),
                      save_figure=True,
                      figure_output_path=None,
                      plot_figure=False,
                      x_ticks_num=4,
                      y_ticks_num=4,
                      cmap="rainbow",
                      dpi=150,
                      return_ax=False
                      ):
    """
    plot heat map
    :param data_matrix: 数据矩阵
    :param x_value_list: 数据矩阵对应的x轴取值列表（输出图像的横轴）
    :param y_value_list: 数据矩阵对应的y轴取值列表（输出图像的纵轴）
    :param draw_size: 绘制大小
    :param title: 绘制标题
    :param xlabel: x轴名称
    :param ylabel: y轴名称
    :param figsize: 画布大小
    :param save_figure: 是否保存输出
    :param figure_output_path: 输出保存路径
    :param plot_figure: 是否输出图像
    :param x_ticks_num: x轴刻度数量（等间距）
    :param y_ticks_num: y轴刻度数量（等间距）
    :param cmap: 颜色映射
    :param dpi: 分辨率
    :param return_ax: 是否返回句柄（用于组合图像）
    :return: None
    """

    assert len(np.shape(data_matrix)) == 2

    raw_shape = np.shape(data_matrix)
    # draw index list
    draw_x_index_list = np.linspace(0, raw_shape[1]-1, min(draw_size[1], raw_shape[1])).astype(np.int32)
    draw_y_index_list = np.linspace(0, raw_shape[0]-1, min(draw_size[0], raw_shape[0])).astype(np.int32)
    # draw data matrix
    draw_data_matrix = data_matrix[draw_y_index_list, :][:, draw_x_index_list]
    # draw x, y value
    draw_x_value_list = np.linspace(np.min(x_value_list), np.max(x_value_list), len(draw_x_index_list))
    draw_y_value_list = np.linspace(np.min(y_value_list), np.max(y_value_list), len(draw_y_index_list))

    # ticks
    x_ticks = np.around(np.linspace(np.min(x_value_list), np.max(x_value_list), x_ticks_num), 1)
    y_ticks = np.around(np.linspace(np.min(y_value_list), np.max(y_value_list), y_ticks_num), 1)

    # ticks position
    x_ticks_position = list()
    y_ticks_position = list()

    for t in x_ticks:
        idx_pos = np.argmin(np.abs(t - draw_x_value_list))
        x_ticks_position.append(idx_pos)

    for t in y_ticks:
        idx_pos = len(draw_y_value_list) - np.argmin(np.abs(t - draw_y_value_list))
        y_ticks_position.append(idx_pos)

    # plot
    plt.figure(figsize=figsize, dpi=dpi)
    ax = sns.heatmap(draw_data_matrix, annot=False, cmap=cmap)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yticks(y_ticks_position)
    ax.set_xticks(x_ticks_position)
    ax.set_title(title)
    ax.set_xticklabels(x_ticks)
    ax.set_yticklabels(y_ticks)

    if save_figure and figure_output_path is not None:
        plt.savefig(figure_output_path)
    if plot_figure:
        plt.show()

    if return_ax:
        return ax
    else:
        return 0


def two_dim_curve_gif_func(data_matrix,
                           x_value_list,
                           y_value_list,
                           figure_size,
                           split_alone_axis,
                           gif_frame_num=None,
                           gif_fps=30,
                           title="",
                           xlabel="x",
                           ylabel="y",
                           x_limits="auto",
                           y_limits="auto",
                           gif_output_path=None,
                           figure_dpi=150
                           ):
    """
    绘制二维曲线的动图
    :param data_matrix: 数据矩阵
    :param x_value_list: 数据矩阵对应的x轴取值列表（输出图像的横轴）
    :param y_value_list: 数据矩阵对应的y轴取值列表（输出图像的横轴）
    :param figure_size: 输出图像大小
    :param split_alone_axis: 选择作为时间轴的分量（0， 1）
    :param gif_frame_num: gif总帧数，默认为全部帧
    :param gif_fps: gif帧率，默认为30帧/s
    :param title: 标题
    :param xlabel: x轴名称
    :param ylabel: y轴名称
    :param x_limits: x轴坐标范围(可传入值、自动模式或者固定模式) "auto", "fixed"
    :param y_limits: y轴坐标范围
    :param gif_output_path: 保存路径
    :param figure_dpi: 输出gif的分辨率
    :return:
    """
    assert len(np.shape(data_matrix)) == 2
    assert split_alone_axis == 0 or split_alone_axis == 1

    total_frame = data_matrix.shape[split_alone_axis]
    if gif_frame_num is not None:
        total_frame = gif_frame_num

    frame_index_list = np.linspace(0, data_matrix.shape[split_alone_axis]-1, total_frame).astype(np.int32)

    # create temp output path
    temp_figure_output_root = os.path.join(os.path.split(gif_output_path)[0], "temp")
    if os.path.exists(temp_figure_output_root):
        shutil.rmtree(temp_figure_output_root)
    os.makedirs(temp_figure_output_root)

    plt.figure(figsize=figure_size, dpi=figure_dpi)

    for i, _frame in enumerate(frame_index_list):
        plt.cla()
        plt.clf()

        if len(y_limits) == 2:
            plt.ylim(y_limits)
        elif y_limits == "auto":
            pass
        elif y_limits == "fixed":
            y_min = np.min(data_matrix)
            y_max = np.max(data_matrix)
            plt.ylim(y_min - (y_max - y_min) * 0.1, y_max + (y_max - y_min) * 0.1)

        if split_alone_axis == 0:
            temp_frame = data_matrix[_frame, :]
            plt.plot(x_value_list, temp_frame)

            if len(x_limits) == 2:
                plt.xlim(x_limits)
            elif x_limits == "auto":
                pass
            elif x_limits == "fixed":
                x_min = np.min(x_value_list)
                x_max = np.max(x_value_list)
                plt.xlim(x_min-(x_max-x_min)*0.1, x_max+(x_max-x_min)*0.1)

        else:
            temp_frame = data_matrix[:, _frame]
            plt.plot(y_value_list, temp_frame)

            if len(x_limits) == 2:
                plt.xlim(x_limits)
            elif x_limits == "auto":
                pass
            elif x_limits == "fixed":
                x_min = np.min(y_value_list)
                x_max = np.max(y_value_list)
                plt.xlim(x_min-(x_max-x_min)*0.1, x_max+(x_max-x_min)*0.1)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title+"_{}".format(_frame))
        plt.savefig(os.path.join(temp_figure_output_root, "{}.png".format(i)), dpi=figure_dpi)

    # create gif
    gif_creator(temp_figure_output_root, gif_output_path, fps=gif_fps)
    shutil.rmtree(temp_figure_output_root)


def gif_creator(temp_root, output_path, fps):
    """
    生成gif
    """
    count_ = len(os.listdir(temp_root))
    with imageio.get_writer(output_path, mode="I", fps=fps) as Writer:
        for ind in range(count_):
            temp_image_path = os.path.join(temp_root, "{}.png".format(ind))
            image = imageio.imread(temp_image_path)
            Writer.append_data(image)


def two_dim_heat_map_gif_func(data_matrix):
    pass


def two_dim_curve_plot_func():
    pass
