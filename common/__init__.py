# coding=utf-8
"""
Author:DYK
Email:y.d@pku.edu.cn

date:27/6/2022 下午8:54
desc:
"""

import click
import ast


class ClickPythonLiteralOption(click.Option):

    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except Exception as e:
            print(e)
            raise click.BadParameter(value)
