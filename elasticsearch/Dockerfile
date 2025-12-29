FROM docker.elastic.co/elasticsearch/elasticsearch:8.19.4

RUN elasticsearch-plugin install --batch \
  pl.allegro.tech.elasticsearch.plugin:elasticsearch-analysis-morfologik:8.19.4

ENV discovery.type=single-node
ENV xpack.security.enabled=false
