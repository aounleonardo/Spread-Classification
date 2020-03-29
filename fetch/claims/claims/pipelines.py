# -*- coding: utf-8 -*-
from scrapy.exceptions import DropItem

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

# TODO move to a config file
ACCEPTED_RATINGS = ['True', 'False']
class ClaimsPipeline(object):
    def process_item(self, item, spider):
        # item.setdefault('initial_tweets', [])
        if item.get('rating') not in ACCEPTED_RATINGS:
            raise DropItem('Unaccepted rating')
        item.setdefault('sysomos_file', "")
        return item
