# -*- coding: utf-8 -*-
import scrapy
from scrapy.loader import ItemLoader

from claims.items import ClaimsItem


class SnopesSpider(scrapy.Spider):
    name = "snopes"
    allowed_domains = ["snopes.com"]

    custom_settings = {"DEPTH_LIMIT": 10}
    start_urls = [
        "http://snopes.com/fact-check/",
    ]

    def parse(self, response):
        for article in response.xpath(
            '//article[@class="media-wrapper"]/a/@href'
        ).getall():
            yield response.follow(
                article, callback=self.parse_article,
            )

        next_page = response.xpath('//a[@class="btn-next btn"]/@href').get()
        if next_page is not None:
            yield response.follow(next_page, callback=self.parse)

    def parse_article(self, response):
        loader = ItemLoader(item=ClaimsItem(), response=response)
        loader.add_xpath("text", '//div[@class="claim"]/p')
        loader.add_xpath("rating", '//h5[starts-with(@class,"rating-label")]')
        loader.add_value("fact_check", response.url)
        yield loader.load_item()
