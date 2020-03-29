# -*- coding: utf-8 -*-
from scrapy import Field
from scrapy.loader.processors import MapCompose, TakeFirst
from w3lib.html import remove_tags

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class ClaimsItem(scrapy.Item):
    # define the fields for your item here like:
    text = Field(
        input_processor=MapCompose(remove_tags),
        output_processor=TakeFirst(),
    )
    rating = Field(
        input_processor=MapCompose(remove_tags),
        output_processor=TakeFirst(),
    )
    fact_check = Field(output_processor=TakeFirst())
    sysomos_file = Field()
