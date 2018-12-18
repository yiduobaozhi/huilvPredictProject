__author__ = '947'
#coding:utf-8
from jpype import *


startJVM(getDefaultJVMPath(), "-Djava.class.path=E:\hanlp-1.6.6-release\hanlp-1.6.6.jar;E:\hanlp-1.6.6-release")
HanLP = JClass('com.hankcs.hanlp.HanLP')

print(HanLP.segment('你好，欢迎在Python中调用HanLP的API'))