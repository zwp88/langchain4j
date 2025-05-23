package dev.langchain4j.agentic;

import dev.langchain4j.agentic.declarative.ExitCondition;
import dev.langchain4j.agentic.declarative.LoopAgent;
import dev.langchain4j.agentic.declarative.SequenceAgent;
import dev.langchain4j.agentic.declarative.Subagent;
import dev.langchain4j.service.V;

import dev.langchain4j.agentic.Agents.CreativeWriter;
import dev.langchain4j.agentic.Agents.AudienceEditor;
import dev.langchain4j.agentic.Agents.StyleEditor;
import dev.langchain4j.agentic.Agents.StyleScorer;

import org.junit.jupiter.api.Test;

import static dev.langchain4j.agentic.Models.BASE_MODEL;
import static org.assertj.core.api.Assertions.assertThat;

public class DeclarativeAgents {

    @SequenceAgent(outputName = "story", subagents = {
            @Subagent(agentClass = CreativeWriter.class, outputName = "story"),
            @Subagent(agentClass = AudienceEditor.class, outputName = "story"),
            @Subagent(agentClass = StyleEditor.class, outputName = "story")
    })
    public interface StoryCreator {

        @Agent
        String write(@V("topic") String topic, @V("style") String style, @V("audience") String audience);
    }

    @Test
    void declarative_sequence_tests() {
        StoryCreator storyCreator = AgentServices.createAgent(StoryCreator.class, BASE_MODEL);

        String story = storyCreator.write("dragons and wizards", "fantasy", "young adults");
        System.out.println(story);
    }

    @LoopAgent(outputName = "story", maxIterations = 5, subagents = {
            @Subagent(agentClass = StyleScorer.class, outputName = "score"),
            @Subagent(agentClass = StyleEditor.class, outputName = "story")
    })
    public interface StyleReviewLoopAgent {

        @Agent
        String write(@V("story") String story);

//        @ExitCondition
//        static boolean exit(Cognisphere cognisphere) {
//            return cognisphere.readState("score", 0.0) >= 0.8;
//        }

        @ExitCondition
        static boolean exit(@V("score") double score) {
            return score >= 0.8;
        }
    }

    @SequenceAgent(outputName = "story", subagents = {
            @Subagent(agentClass = CreativeWriter.class, outputName = "story"),
            @Subagent(agentClass = StyleReviewLoopAgent.class, outputName = "story"),
    })
    public interface StoryCreatorWithReview {

        @Agent
        String write(@V("topic") String topic, @V("style") String style);
    }

    @Test
    void declarative_sequence_and_loop_tests() {
        StoryCreatorWithReview storyCreator = AgentServices.createAgent(StoryCreatorWithReview.class, BASE_MODEL);

        String story = storyCreator.write("dragons and wizards", "comedy");
        System.out.println(story);

        Cognisphere cognisphere = ((CognisphereOwner) storyCreator).cognisphere();
        assertThat(cognisphere.readState("topic")).isEqualTo("dragons and wizards");
        assertThat(cognisphere.readState("style")).isEqualTo("comedy");
        assertThat(story).isEqualTo(cognisphere.readState("story"));
        assertThat(cognisphere.readState("score", 0.0)).isGreaterThanOrEqualTo(0.8);
    }
}
