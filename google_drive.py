#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 17:42:39 2020

@author: yuichiro
"""
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()
gauth.LocalWebserverAuth()

drive = GoogleDrive(gauth)


file_list = drive.ListFile(
    {'q': "'11mHAi7ITvL3lB6HFNhFj0NscinsvQ1N8' in parents"}).GetList()
for f in file_list:
  # 3. Create & download by id.
  print('title: %s, id: %s' % (f['title'], f['id']))

# lab_folder_id = drive.ListFile(\
# {'q': "'root' in parents and title= 'lab' and trashed=False and mimeType = \
# 'application/vnd.google-apps.folder' "}).GetList()[0]['id']

# file_list = drive.ListFile(
#     {'q': """'{folder_id}' in parents and trashed=False""".format(folder_id=lab_folder_id)}).GetList()
# [ (file['title'],file['id'] ) for file in file_list]