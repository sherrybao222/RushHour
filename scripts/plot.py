'''
Script for plotting, py27
'''
import os, sys, cv2
import numpy as np
from graphviz import Digraph
from datetime import datetime
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt

def plot_tree(root, highlighted_node=None, updated_node=[], decision_node=None, highlighted_edges=[]):
	# highlighted_edges=[] is a list of Node
	graph_attr={'fixedsize':'false', 
				# 'size':'12,12',
				# 'bgcolor':'transparent',
				'resolution':'100'}
	node_attr = {'align':'top',
						'style':'filled',
						'fillcolor':'gray',
						'fontsize':'12',
						'ranksep':'0.1',
						'height':'0.4',
						'pad':'0.212,0.055',
						# 'autosize':'false', 
						# 'fixedsize':'true',
						'size':'3!'}
	dot = Digraph(graph_attr=graph_attr,node_attr=node_attr)
	if root == decision_node:
		dot.node(name=str(root), label='', pos='0,5!', fillcolor='red', penwidth='4', color='red')
	elif decision_node != None:
		dot.node(name=str(root), label='', pos='0,5!', penwidth='4', color='red', fillcolor='gray')
	elif root in updated_node:
		dot.node(name=str(root), label='', pos='0,5!', fillcolor='lightblue', penwidth='4', color='lightblue')
	elif root == highlighted_node:
		dot.node(name=str(root), label='', pos='0,5!', fillcolor='blue', penwidth='4', color='blue')
	elif root in highlighted_edges:
		dot.node(name=str(root), label='', pos='0,5!', penwidth='4', color='blue', fillcolor='gray')
	else:
		dot.node(name=str(root), label='', pos='0,5!')
	queue = []
	queue.append(root)
	while queue: #BFS
		n = queue.pop(0)
		for child in n.children:
			if child == highlighted_node:
				dot.node(name=str(child), label='', fillcolor='blue', penwidth='4', color='blue')
			elif child in updated_node:
				dot.node(name=str(child), label='', fillcolor='lightblue', penwidth='4', color='lightblue')
			elif child == decision_node:
				dot.node(name=str(child), label='', fillcolor='red', penwidth='4', color='red')
				dot.edge(str(n), str(child), penwidth='4', color='red')
				queue.append(child)
				continue
			elif child in highlighted_edges:
				dot.node(name=str(child), label='', penwidth='4', color='blue', fillcolor='gray')
			else:
				dot.node(name=str(child), label='')
			if child in highlighted_edges:
				dot.edge(str(n), str(child), penwidth='4', color='blue')
			else:
				dot.edge(str(n), str(child))
			queue.append(child)
	curtime = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
	dot.render('/Users/chloe/Desktop/'+curtime,view=False, cleanup=True, format='png')
	img = Image.open('/Users/chloe/Desktop/'+curtime+'.png')
	background = Image.open('/Users/chloe/Desktop/zbackground.png')
	background.paste(img, ((background.size[0]-img.size[0])/2, 220),img)
	background.save('/Users/chloe/Desktop/resized_'+curtime+'.png','PNG')
	os.remove('/Users/chloe/Desktop/'+curtime+'.png')
	return '/Users/chloe/Desktop/resized_'+curtime+'.png'

def plot_board(node):
	''' visualize the current board configuration '''
	matrix = node.board_matrix()
	cmap = plt.cm.Set1
	cmap.set_bad(color='white')
	fig, ax = plt.subplots(figsize=(16,12))
	ax.set_xticks(np.arange(-0.5, 5, 1))
	ax.set_yticks(np.arange(-0.5, 5, 1))
	ax.set_axisbelow(True)
	ax.grid(b=True, which='major',color='gray', linestyle='-', linewidth=1, alpha=0.1)
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	for tic in ax.xaxis.get_major_ticks():
		tic.tick1On = tic.tick2On = False
	for tic in ax.yaxis.get_major_ticks():
		tic.tick1On = tic.tick2On = False
	im = ax.imshow(matrix, cmap=cmap)
	show_car_label = False
	if show_car_label:
		for i in range(len(matrix)):
			for j in range(len(matrix[i])):
				num = matrix[i, j]
				if num == 0:
					num = 'R'
				elif num > 0:
					num -= 1
				else:
					num = ''
				text = ax.text(j, i, num, ha="center", va="center", color="black", fontsize=36)
	for axis in ['top','bottom','left','right']:
		ax.spines[axis].set_linewidth(4)
		ax.spines[axis].set_zorder(0)
	fig.patch.set_facecolor('white')
	fig.patch.set_alpha(0.2)
	curtime = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
	plt.savefig('/Users/chloe/Desktop/board_'+curtime+'.png', 
			facecolor = fig.get_facecolor(), transparent = True)
	plt.close()
	return '/Users/chloe/Desktop/board_'+curtime+'.png'


def plot_board_and_tree(root, board_node, highlighted_node=None, 
					updated_node=[], decision_node=None, highlighted_edges=[], 
					text='', text2=''):
	''' plot board and tree in the same image '''
	tree_filename = plot_tree(root, highlighted_node, updated_node, decision_node, highlighted_edges)
	board_filename = plot_board(board_node)
	final_img = concatenate_images_h(Image.open(board_filename), Image.open(tree_filename))
	draw = ImageDraw.Draw(final_img)
	font = ImageFont.truetype("Verdana.ttf", 100)
	draw.text((final_img.size[0]/2-100, 30), text, (0,0,0), font=font)
	draw.text((400, 30), text2, (0,0,0), font=font)
	curtime = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
	final_filename = '/Users/chloe/Desktop/RH/zip_'+curtime+'.png'
	final_img.save(final_filename)
	os.remove(tree_filename)
	os.remove(board_filename)


def concatenate_images_h(im1, im2):
	''' concatenate two images one next to another horizontally '''
	dst = Image.new('RGB', (im1.width + im2.width, im1.height))
	dst.paste(im1, (0, 0))
	dst.paste(im2, (im1.width, 0))
	return dst

def make_movie(path='/Users/chloe/Desktop/RH/'):
	''' make a movie using png files '''
	os.chdir(path)
	image_folder = path
	video_name = 'BFS_Movie.avi' 
	images = sorted(filter(os.path.isfile, [x for x in os.listdir(path) if x.endswith('png')]), key=lambda p: os.path.exists(p) and os.stat(p).st_mtime or time.mktime(datetime.now().timetuple()))
	frame = cv2.imread(os.path.join(image_folder, images[0]))
	height, width, layers = frame.shape
	video = cv2.VideoWriter(video_name, 0, 2, (width,height))
	for image in images:
		resized=cv2.resize(cv2.imread(os.path.join(image_folder, image)), (width, height)) 
		video.write(resized)
	cv2.destroyAllWindows()
	video.release()

# make_movie()


# img = Image.new('RGB', (4500, 1200), (255, 255, 255))
# img.save('/Users/chloe/Desktop/zbackground.png')

