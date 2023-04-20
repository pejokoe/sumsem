from classes import *


a1 = []
a1.append(Song("Song1", "Singer1", 120))
a1.append(Song("Song2", "Singer1", 129))
a1.append(Song("Song3", "Singer2", 128))
a1.append(Song("Song4", "Singer2", 127))
a1.append(Song("Song5", "Singer2", 126))
a1.append(Song("Song12", "Singer3", 125))
a1.append(Song("Song13", "Singer3", 124))
a1.append(Song("Song14", "Singer3", 123))
A1 = Album("Title1")
for song in a1:
    A1.addSong(song)

a2 = []
a2.append(Song("Song11", "Singer1", 120))
a2.append(Song("Song22", "Singer1", 121))
a2.append(Song("Song33", "Singer1", 122))
a2.append(Song("Song44", "Singer3", 123))
a2.append(Song("Song55", "Singer4", 124))
a2.append(Song("Song66", "Singer4", 125))
a2.append(Song("Song77", "Singer5", 126))
a2.append(Song("Song88", "Singer5", 127))
A2 = Album("Title2")
for song in a2:
    A2.addSong(song)

library = Library()
library.addAlbum(A1)
library.addAlbum(A2)
print(library.retrieve("Singer3"))
