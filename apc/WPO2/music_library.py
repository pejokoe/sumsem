#!/usr/bin/python3

class Song():
    def __init__(self, title, artist, duration):
        self._title = title
        self._artist = artist
        self._duration = duration
    def getArtist(self):
        return self._artist
    def getTitle(self):
        return self._title

class Album():
    def __init__(self, title):
        self._title = title
        self._songs = []
    def addSong(self, song):
        if isinstance(song, Song):
            self._songs.append(song)
        else:
            print("AddSong only possible with a song object!")

    def retrieve(self, artist):
        retrieved = []
        for song in self._songs:
            if song.getArtist() == artist:
                retrieved.append((song.getTitle(), self._title))
        return retrieved

class Library():
    def __init__(self):
        self._albums = []

    def addAlbum(self, album):
        if isinstance(album, Album):
            self._albums.append(album)

    def retrieve(self, artist):
        retrieved = []
        for album in self._albums:
             songList = album.retrieve(artist)
             if songList:
                 retrieved.extend(songList)
        return retrieved


