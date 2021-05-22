#!/usr/bin/env python3
"""Generates the TensorStore logo in SVG format."""

import argparse

import numpy as np


def write_logo(path: str) -> None:

  letter_cells = np.array([
      [1, 1, 1, 1, 1, 0, 0, 0, 0],
      [0, 0, 1, 0, 0, 0, 0, 0, 0],
      [0, 0, 1, 0, 0, 1, 1, 1, 0],
      [0, 0, 1, 0, 1, 0, 0, 0, 1],
      [0, 0, 1, 0, 0, 1, 0, 0, 0],
      [0, 0, 1, 0, 0, 0, 1, 0, 0],
      [0, 0, 1, 0, 0, 0, 0, 1, 0],
      [0, 0, 0, 0, 1, 0, 0, 0, 1],
      [0, 0, 0, 0, 0, 1, 1, 1, 0],
  ], dtype=bool)

  width_over_height = letter_cells.shape[1] / letter_cells.shape[0]

  margin = 2

  base_size = 128

  screen_size = np.array(
      [width_over_height * base_size + 2 * margin, base_size + 2 * margin])

  cell_size = base_size / letter_cells.shape[0]

  with open(path, 'w') as f:
    f.write(f'<svg xmlns="http://www.w3.org/2000/svg" ' +
            f'viewBox="0 0 {screen_size[0]} {screen_size[1]}">')

    grid_line_width = 1.5
    grid_line_color = fill_color = 'currentColor'
    cell_margin = 1.5

    # Draw horizontal grid lines
    for i in range(letter_cells.shape[0] + 1):
      f.write(f'<line stroke="{grid_line_color}" ' +
              f'stroke-width="{grid_line_width}" ' +
              f'x1="{margin-grid_line_width/2}" ' +
              f'y1="{round(margin+i*cell_size,1)}" ' +
              f'x2="{round(screen_size[0]-margin+grid_line_width/2,1)}" ' +
              f'y2="{round(margin+i*cell_size,1)}"/>')

    # Draw vertical grid lines
    for i in range(letter_cells.shape[1] + 1):
      f.write(f'<line stroke="{grid_line_color}" ' +
              f'stroke-width="{grid_line_width}" ' +
              f'y1="{margin}" x1="{round(margin+i*cell_size,1)}" ' +
              f'y2="{round(screen_size[1]-margin,1)}" ' +
              f'x2="{round(margin+i*cell_size,1)}"/>')

    for y in range(letter_cells.shape[0]):
      for x in range(letter_cells.shape[1]):
        if not letter_cells[y, x]:
          continue
        f.write(f'<rect fill="{fill_color}" ' +
                f'x="{round(margin+x*cell_size+cell_margin,1)}" ' +
                f'y="{round(margin+y*cell_size+cell_margin,1)}" ' +
                f'width="{round(cell_size-2*cell_margin,1)}" ' +
                f'height="{round(cell_size-2*cell_margin, 1)}"/>')
    f.write('</svg>\n')


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('output')

  args = parser.parse_args()
  write_logo(args.output)


if __name__ == '__main__':
  main()
