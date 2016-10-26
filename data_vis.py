import sys,csv
import operator
import numpy as np
from sklearn.decomposition import NMF
import time
def no_of_char(chars):

	li = []
	for i in chars:
		li.append(','.join([ x for x in iter(set(chars[i]).difference(set(li)))]))
	print len(li)

def info_train_txt(fread):
	data = file.read(file(fread))
	rows = data.split('\n')
	ques_to_exp = {}
	exp_to_ques = {}
	#print max(exp_to_ques.iteritems(), key=operator.itemgetter(1))
	#print exp_to_ques[max(exp_to_ques)]
	for datas in rows:
		line  = datas.split('\t')
		if len(line)<2:
			break
		if line[1] in exp_to_ques:
			exp_to_ques[line[1]] +=1
		else:
			exp_to_ques[line[1]] =1
	with open('exp_to_ques.txt','wb') as csvfile:
		#row = csv.writer(csvfile,delimiter=' ')
		for i in exp_to_ques:
			print i +'    '+str(exp_to_ques[i])
			csvfile.write(i+'\t'+str(exp_to_ques[i])+'\n')

def matrix_factorization(R, P, Q, K, steps, alpha=0.0002, beta=0.02):
	Q = Q.T
	for step in xrange(steps):
		for i in xrange(len(R)):
			#print i
			for j in xrange(len(R[i])):
				if R[i][j] > 0:
					eij = R[i][j] - np.dot(P[i,:],Q[:,j])
					for k in xrange(K):
						P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
						Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
		eR = np.dot(P,Q)
		e = 0
		for i in xrange(len(R)):
			for j in xrange(len(R[i])):
				if R[i][j] > 0:
					e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
					for k in xrange(K):
						e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
		if e < 0.001:
			break
	return P, Q.T

def get_lines(fread):
	data = file.read(file(fread))
	rows = data.split('\n')
	return rows
def get_QE_dict(rows):
	new_dict = {}
	ques = []
	experts_list =  []
	experts= []

	for datas in rows:
		line  = datas.split('\t')
		if len(line)<3:
			break
		if line[1] not in experts:
			experts.append(line[1])
		if line[0] in new_dict:
			if line[1] in new_dict[line[0]]:
				 new_dict[line[0]][line[1]].append(int(line[2]))
			else:
				new_dict[line[0]][line[1]] = []
				new_dict[line[0]][line[1]].append(int(line[2]))
		else:
			new_dict[line[0]]={}
			ques.append(line[0])
			new_dict[line[0]][line[1]] = []
			new_dict[line[0]][line[1]].append(int(line[2]))
	print 'QE exit'
	return new_dict,ques,experts

def get_prob_matrix(fread):
	maps_score,ques,experts =  get_QE_dict(fread)
	matrix  =  np.zeros((len(ques),len(experts)))
	for i in maps_score:
		for j in maps_score[i]:
			zero = 0.0;
			one = 0.0;
			for k in maps_score[i][j]:
				if k == 0:
					zero+=1.0
				else:
					one+=1.0
			if one == 0:
				maps_score[i][j] = 0.0
			else:
				maps_score[i][j] = float((one)/(one+zero))
			matrix[ques.index(i)][experts.index(j)] = maps_score[i][j]
	print 'get prob matrix'
	return matrix,ques,experts


def write_prob_matrix(prob_score):
	with open('prob_score1.txt','wb') as csvfile:
		for i in prob_score:
			for j in prob_score[i]:
				csvfile.write(i+','+j+','+str(prob_score[i][j])+'\n')

def print_output(matrix, ques, expert):
	rows = get_lines('validate_nolabel.txt')[1:]
	with open('output.txt','wb') as out:
		out.write('qid,uid,label\n')
		for i in rows:
			data = i.split(',')
			if data[0] not in ques or data[1].splitlines()[0] not in expert:
				val = 0.0
			else:
				val = matrix[ques.index(data[0])][expert.index(data[1].splitlines()[0])]
			out.write(i.splitlines[0]+','+str(val)+'\n')


def get_matrix_fact(fread,steps):
	matrix,ques,experts = get_prob_matrix(get_lines(fread))
	print 'MATRIX FACT'
	N = len(matrix)
	M = len(matrix[0])
	K = 2
	P = np.random.rand(N,K)
	Q = np.random.rand(M,K)
	nP,nQ = matrix_factorization(matrix,P,Q,K,steps)
	nR = np.dot(nP,nQ.T)
	print 'EXIT'
	print_output(nR,ques,experts)
	return nR

def question_info(fread):
	data = file.read(file(fread))
	rows = data.split('\n')
	print len(rows)

def expert_cat_word_char_sequence(fread):
	rows = get_lines(fread)
	expert_interests = {}
	expert_word_sequence = {}
	expert_char_sequence = {}
	i = 0
	print len(rows)
	for datas in rows:
		line = datas.split('\t')
		#print len(line)
		if len(line)<4:
			print line
			continue
		categories = line[1].split('\\')
		expert_interests[line[0]] = categories 
		expert_word_sequence[line[0]] = line[2].split('/')
		expert_char_sequence[line[0]] = line[3].split('/')		
	return expert_interests,expert_word_sequence,expert_char_sequence


if __name__=="__main__":
	#split_user_info('user_info.txt')
	#question_info('question_info.txt')
	#info_train_txt('invited_info_train.txt')
	t1 = time.clock()
	get_matrix_fact('invited_info_train.txt',10)
	print str(time.clock()-t1)