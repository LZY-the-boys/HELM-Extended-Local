import Scenario from "@/types/Scenario";
import getBenchmarkEndpoint from "@/utils/getBenchmarkEndpoint";
import getBenchmarkSuite from "@/utils/getBenchmarkSuite";

export default async function getScenarioByName(
  scenarioName: string,
  signal: AbortSignal,
): Promise<Scenario | undefined> {
  try {
    const scenario = await fetch(
      getBenchmarkEndpoint(
        `/benchmark_output/runs/${getBenchmarkSuite()}/${scenarioName}/scenario.json`,
      ),
      { signal },
    );

    return (await scenario.json()) as Scenario;
  } catch (error) {
    console.log(error);
    return undefined;
  }
}
