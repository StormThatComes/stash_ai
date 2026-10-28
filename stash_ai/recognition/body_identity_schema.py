"""Derived body identity schema, independent of the matching implementation."""
SCHEMA='''
CREATE TABLE body_members (
 observation TEXT PRIMARY KEY REFERENCES body_observations(id),
 person TEXT REFERENCES people(id), role TEXT NOT NULL CHECK(role IN ('anchor','auxiliary')),
 status TEXT NOT NULL CHECK(status IN ('accepted','provisional','withheld')),
 policy TEXT NOT NULL, proof TEXT NOT NULL);
CREATE INDEX body_members_person ON body_members(person,status);
CREATE TABLE body_origins (
 observation TEXT PRIMARY KEY REFERENCES body_observations(id),
 person TEXT NOT NULL REFERENCES people(id));
'''
