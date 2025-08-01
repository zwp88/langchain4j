package dev.langchain4j.model.mistralai;

import static dev.langchain4j.internal.RetryUtils.withRetryMappingExceptions;
import static dev.langchain4j.internal.Utils.getOrDefault;
import static dev.langchain4j.internal.ValidationUtils.ensureNotBlank;
import static dev.langchain4j.model.mistralai.internal.mapper.MistralAiMapper.tokenUsageFrom;
import static dev.langchain4j.spi.ServiceHelper.loadFactories;
import static java.util.stream.Collectors.toList;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.http.client.HttpClientBuilder;
import dev.langchain4j.model.embedding.DimensionAwareEmbeddingModel;
import dev.langchain4j.model.mistralai.internal.api.MistralAiEmbeddingRequest;
import dev.langchain4j.model.mistralai.internal.api.MistralAiEmbeddingResponse;
import dev.langchain4j.model.mistralai.internal.client.MistralAiClient;
import dev.langchain4j.model.mistralai.spi.MistralAiEmbeddingModelBuilderFactory;
import dev.langchain4j.model.output.Response;
import java.time.Duration;
import java.util.List;

/**
 * Represents a Mistral AI embedding model, such as mistral-embed.
 * You can find description of parameters <a href="https://docs.mistral.ai/api/#operation/createEmbedding">here</a>.
 */
public class MistralAiEmbeddingModel extends DimensionAwareEmbeddingModel {

    private static final String EMBEDDINGS_ENCODING_FORMAT = "float";

    private final MistralAiClient client;
    private final String modelName;
    private final Integer maxRetries;

    public MistralAiEmbeddingModel(MistralAiEmbeddingModelBuilder builder) {
        this.client = MistralAiClient.builder()
                .httpClientBuilder(builder.httpClientBuilder)
                .baseUrl(getOrDefault(builder.baseUrl, "https://api.mistral.ai/v1"))
                .apiKey(builder.apiKey)
                .timeout(builder.timeout)
                .logRequests(getOrDefault(builder.logRequests, false))
                .logResponses(getOrDefault(builder.logResponses, false))
                .build();
        this.modelName = ensureNotBlank(builder.modelName, "modelName");
        this.maxRetries = getOrDefault(builder.maxRetries, 2);
    }

    /**
     * Embeds a list of text segments using the Mistral AI embedding model.
     *
     * @param textSegments the list of text segments to embed
     * @return a Response object containing the embeddings and token usage information
     */
    @Override
    public Response<List<Embedding>> embedAll(List<TextSegment> textSegments) {

        MistralAiEmbeddingRequest request = MistralAiEmbeddingRequest.builder()
                .model(modelName)
                .input(textSegments.stream().map(TextSegment::text).collect(toList()))
                .encodingFormat(EMBEDDINGS_ENCODING_FORMAT)
                .build();

        MistralAiEmbeddingResponse response = withRetryMappingExceptions(() -> client.embedding(request), maxRetries);

        List<Embedding> embeddings = response.getData().stream()
                .map(mistralAiEmbedding -> Embedding.from(mistralAiEmbedding.getEmbedding()))
                .collect(toList());

        return Response.from(embeddings, tokenUsageFrom(response.getUsage()));
    }

    public static MistralAiEmbeddingModelBuilder builder() {
        for (MistralAiEmbeddingModelBuilderFactory factory :
                loadFactories(MistralAiEmbeddingModelBuilderFactory.class)) {
            return factory.get();
        }
        return new MistralAiEmbeddingModelBuilder();
    }

    public static class MistralAiEmbeddingModelBuilder {

        private String baseUrl;
        private String apiKey;
        private String modelName;
        private Duration timeout;
        private Boolean logRequests;
        private Boolean logResponses;
        private Integer maxRetries;
        private HttpClientBuilder httpClientBuilder;

        public MistralAiEmbeddingModelBuilder() {}

        public MistralAiEmbeddingModelBuilder modelName(String modelName) {
            this.modelName = modelName;
            return this;
        }

        public MistralAiEmbeddingModelBuilder modelName(MistralAiEmbeddingModelName modelName) {
            this.modelName = modelName.toString();
            return this;
        }

        /**
         * @param baseUrl      the base URL of the Mistral AI API. It use a default value if not specified
         * @return {@code this}.
         */
        public MistralAiEmbeddingModelBuilder baseUrl(String baseUrl) {
            this.baseUrl = baseUrl;
            return this;
        }

        /**
         * @param apiKey       the API key for authentication
         * @return {@code this}.
         */
        public MistralAiEmbeddingModelBuilder apiKey(String apiKey) {
            this.apiKey = apiKey;
            return this;
        }

        /**
         * @param timeout      the timeout duration for API requests. It uses a default value of 60 seconds if not specified
         *                     <p>
         *                     The default value is 60 seconds
         * @return {@code this}.
         */
        public MistralAiEmbeddingModelBuilder timeout(Duration timeout) {
            this.timeout = timeout;
            return this;
        }

        /**
         * @param logRequests  a flag indicating whether to log API requests
         * @return {@code this}.
         */
        public MistralAiEmbeddingModelBuilder logRequests(Boolean logRequests) {
            this.logRequests = logRequests;
            return this;
        }

        /**
         * @param logResponses a flag indicating whether to log API responses
         * @return {@code this}.
         */
        public MistralAiEmbeddingModelBuilder logResponses(Boolean logResponses) {
            this.logResponses = logResponses;
            return this;
        }

        /**
         * @return {@code this}.
         */
        public MistralAiEmbeddingModelBuilder maxRetries(Integer maxRetries) {
            this.maxRetries = maxRetries;
            return this;
        }

        /**
         * @param httpClientBuilder the HTTP client builder to use for creating the HTTP client
         * @return {@code this}.
         */
        public MistralAiEmbeddingModelBuilder httpClientBuilder(HttpClientBuilder httpClientBuilder) {
            this.httpClientBuilder = httpClientBuilder;
            return this;
        }

        public MistralAiEmbeddingModel build() {
            return new MistralAiEmbeddingModel(this);
        }
    }
}
